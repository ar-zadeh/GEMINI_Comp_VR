#include "cvisionserver.h"
#include "driverlog.h"
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <chrono>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace
{
    static const uint32_t kSharedFrameMagic = 0x4F455946; // 'OEYF'
    static const uint32_t kSharedFrameVersion = 1;
    static const wchar_t* kSharedFrameMapName = L"Local\\OpenEyeVRFrameMap_v1";
    static const uint32_t kDefaultSharedFrameCapacity = 16 * 1024 * 1024; // 16 MiB JPEG payload.
    static const int kAsyncCaptureFps = 30;

    // Base64 encoding table.
    static const char* s_base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
}

CVisionServer* g_pVisionServer = nullptr;

CVisionServer::CVisionServer()
    : m_bCapturing(false)
    , m_bInitialized(false)
    , m_bAsyncCaptureRunning(false)
    , m_pAsyncCaptureThread(nullptr)
    , m_bHasLastFrame(false)
    , m_sharedFrameCapacity(kDefaultSharedFrameCapacity)
#if defined(_WIN32)
    , m_hFrameMap(nullptr)
    , m_pFrameMapView(nullptr)
#endif
{
}

CVisionServer::~CVisionServer()
{
    Shutdown();
}

bool CVisionServer::Initialize(int windowX, int windowY, int windowWidth, int windowHeight)
{
    if (m_bInitialized)
    {
        Shutdown();
    }

    if (!m_frameCapture.Initialize(windowX, windowY, windowWidth, windowHeight))
    {
        DriverLog("CVisionServer: Failed to initialize frame capture\n");
        return false;
    }

    if (!InitSharedMemory())
    {
        DriverLog("CVisionServer: Failed to initialize shared memory publisher\n");
        m_frameCapture.Shutdown();
        return false;
    }

    StartAsyncCapture();
    m_bInitialized = true;

    DriverLog("CVisionServer: Initialized for window region (%d,%d) %dx%d\n",
        windowX, windowY, windowWidth, windowHeight);
    return true;
}

void CVisionServer::Shutdown()
{
    StopAsyncCapture();
    ShutdownSharedMemory();
    m_frameCapture.Shutdown();
    m_bHasLastFrame = false;
    m_bInitialized = false;
}

void CVisionServer::StartAsyncCapture()
{
    if (m_bAsyncCaptureRunning)
    {
        return;
    }

    m_bAsyncCaptureRunning = true;
    m_pAsyncCaptureThread = new std::thread(&CVisionServer::AsyncCaptureThread, this);
}

void CVisionServer::StopAsyncCapture()
{
    m_bAsyncCaptureRunning = false;

    if (m_pAsyncCaptureThread)
    {
        if (m_pAsyncCaptureThread->joinable())
        {
            m_pAsyncCaptureThread->join();
        }
        delete m_pAsyncCaptureThread;
        m_pAsyncCaptureThread = nullptr;
    }
}

bool CVisionServer::InitSharedMemory()
{
#if defined(_WIN32)
    const uint64_t totalSize = static_cast<uint64_t>(sizeof(SharedFrameMeta)) + static_cast<uint64_t>(m_sharedFrameCapacity);
    if (totalSize > 0xFFFFFFFFull)
    {
        DriverLog("CVisionServer: Shared memory size too large: %llu\n", totalSize);
        return false;
    }

    m_hFrameMap = CreateFileMappingW(
        INVALID_HANDLE_VALUE,
        nullptr,
        PAGE_READWRITE,
        0,
        static_cast<DWORD>(totalSize),
        kSharedFrameMapName);
    if (!m_hFrameMap)
    {
        DriverLog("CVisionServer: CreateFileMappingW failed (%lu)\n", GetLastError());
        return false;
    }

    m_pFrameMapView = static_cast<unsigned char*>(MapViewOfFile(
        m_hFrameMap,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        static_cast<SIZE_T>(totalSize)));
    if (!m_pFrameMapView)
    {
        DriverLog("CVisionServer: MapViewOfFile failed (%lu)\n", GetLastError());
        CloseHandle(m_hFrameMap);
        m_hFrameMap = nullptr;
        return false;
    }

    SharedFrameMeta* meta = reinterpret_cast<SharedFrameMeta*>(m_pFrameMapView);
    memset(meta, 0, sizeof(SharedFrameMeta));
    meta->magic = kSharedFrameMagic;
    meta->version = kSharedFrameVersion;
    meta->headerSize = static_cast<uint32_t>(sizeof(SharedFrameMeta));
    meta->capacity = m_sharedFrameCapacity;
    meta->sequence = 0;

    DriverLog("CVisionServer: Shared memory ready: %ls (capacity=%u bytes)\n", kSharedFrameMapName, m_sharedFrameCapacity);
    return true;
#else
    return true;
#endif
}

void CVisionServer::ShutdownSharedMemory()
{
#if defined(_WIN32)
    if (m_pFrameMapView)
    {
        UnmapViewOfFile(m_pFrameMapView);
        m_pFrameMapView = nullptr;
    }

    if (m_hFrameMap)
    {
        CloseHandle(m_hFrameMap);
        m_hFrameMap = nullptr;
    }
#endif
}

bool CVisionServer::PublishFrameToSharedMemory(const FrameData& frame)
{
#if defined(_WIN32)
    if (!m_pFrameMapView || frame.jpegData.empty())
    {
        return false;
    }

    if (frame.jpegData.size() > static_cast<size_t>(m_sharedFrameCapacity))
    {
        DriverLog("CVisionServer: Dropping frame too large for shared map (%zu > %u)\n", frame.jpegData.size(), m_sharedFrameCapacity);
        return false;
    }

    SharedFrameMeta* meta = reinterpret_cast<SharedFrameMeta*>(m_pFrameMapView);
    unsigned char* payload = m_pFrameMapView + sizeof(SharedFrameMeta);

    uint32_t writeSequence = meta->sequence + 1;
    if ((writeSequence & 1u) == 0)
    {
        writeSequence += 1;
    }

    meta->sequence = writeSequence;
    MemoryBarrier();

    memcpy(payload, frame.jpegData.data(), frame.jpegData.size());
    meta->timestampMs = frame.timestamp;
    meta->width = static_cast<uint32_t>(frame.width);
    meta->height = static_cast<uint32_t>(frame.height);
    meta->jpegSize = static_cast<uint32_t>(frame.jpegData.size());

    MemoryBarrier();
    meta->sequence = writeSequence + 1;
    return true;
#else
    (void)frame;
    return true;
#endif
}

void CVisionServer::AsyncCaptureThread()
{
    const int frameIntervalMs = 1000 / kAsyncCaptureFps;

    while (m_bAsyncCaptureRunning)
    {
        const auto frameStart = std::chrono::steady_clock::now();

        FrameData frame;
        if (m_frameCapture.CaptureFrame(frame))
        {
            PublishFrameToSharedMemory(frame);

            std::lock_guard<std::mutex> lock(m_lastFrameMutex);
            m_lastFrame = std::move(frame);
            m_bHasLastFrame = true;
        }

        const auto frameEnd = std::chrono::steady_clock::now();
        const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart).count();
        if (elapsedMs < frameIntervalMs)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(frameIntervalMs - elapsedMs));
        }
    }
}

std::string CVisionServer::Base64Encode(const std::vector<uint8_t>& data)
{
    std::string result;
    result.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    while (i < data.size())
    {
        uint32_t octet_a = i < data.size() ? data[i++] : 0;
        uint32_t octet_b = i < data.size() ? data[i++] : 0;
        uint32_t octet_c = i < data.size() ? data[i++] : 0;

        uint32_t triple = (octet_a << 16) + (octet_b << 8) + octet_c;

        result += s_base64_chars[(triple >> 18) & 0x3F];
        result += s_base64_chars[(triple >> 12) & 0x3F];
        result += s_base64_chars[(triple >> 6) & 0x3F];
        result += s_base64_chars[triple & 0x3F];
    }

    size_t mod = data.size() % 3;
    if (mod == 1)
    {
        result[result.size() - 1] = '=';
        result[result.size() - 2] = '=';
    }
    else if (mod == 2)
    {
        result[result.size() - 1] = '=';
    }

    return result;
}

std::string CVisionServer::BuildJsonResponse(const VisionResponse& response)
{
    std::ostringstream json;
    json << "{\"type\":\"" << response.type << "\"";

    if (!response.message.empty())
    {
        json << ",\"message\":\"" << response.message << "\"";
    }

    json << ",\"width\":" << response.width;
    json << ",\"height\":" << response.height;
    json << ",\"frameCount\":" << response.frameCount;

    if (!response.frames.empty())
    {
        json << ",\"frames\":[";
        for (size_t i = 0; i < response.frames.size(); i++)
        {
            if (i > 0) json << ",";
            json << "\"" << response.frames[i] << "\"";
        }
        json << "]";
    }

    json << "}";
    return json.str();
}

bool CVisionServer::HandleCaptureFrame(std::string& jsonResponse)
{
    FrameData frame;
    {
        std::lock_guard<std::mutex> lock(m_lastFrameMutex);
        if (m_bHasLastFrame)
        {
            frame = m_lastFrame;
        }
    }

    if (frame.jpegData.empty())
    {
        if (!m_frameCapture.CaptureFrame(frame))
        {
            VisionResponse resp;
            resp.type = "error";
            resp.message = "Failed to capture frame";
            resp.width = 0;
            resp.height = 0;
            resp.frameCount = 0;
            jsonResponse = BuildJsonResponse(resp);
            return false;
        }

        PublishFrameToSharedMemory(frame);
        {
            std::lock_guard<std::mutex> lock(m_lastFrameMutex);
            m_lastFrame = frame;
            m_bHasLastFrame = true;
        }
    }

    VisionResponse resp;
    resp.type = "frame";
    resp.width = frame.width;
    resp.height = frame.height;
    resp.frameCount = 1;
    resp.frames.push_back(Base64Encode(frame.jpegData));

    jsonResponse = BuildJsonResponse(resp);
    DriverLog("CVisionServer: Returned cached frame %dx%d, %zu bytes\n",
        frame.width, frame.height, frame.jpegData.size());
    return true;
}

bool CVisionServer::HandleCaptureVideo(float duration, int fps, std::string& jsonResponse)
{
    std::lock_guard<std::mutex> lock(m_captureMutex);

    if (m_bCapturing)
    {
        VisionResponse resp;
        resp.type = "error";
        resp.message = "Already capturing";
        resp.width = 0;
        resp.height = 0;
        resp.frameCount = 0;
        jsonResponse = BuildJsonResponse(resp);
        return false;
    }

    m_bCapturing = true;
    DriverLog("CVisionServer: Starting video capture for %.1f seconds at %d fps\n", duration, fps);

    std::vector<FrameData> frames;
    int frameInterval = 1000 / fps;
    int totalFrames = static_cast<int>(duration * fps);

    for (int i = 0; i < totalFrames; i++)
    {
        auto frameStart = std::chrono::steady_clock::now();

        FrameData frame;
        if (m_frameCapture.CaptureFrame(frame))
        {
            frames.push_back(frame);
            PublishFrameToSharedMemory(frame);

            std::lock_guard<std::mutex> lastLock(m_lastFrameMutex);
            m_lastFrame = frame;
            m_bHasLastFrame = true;
        }

        auto frameEnd = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart).count();

        if (elapsed < frameInterval)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(frameInterval - elapsed));
        }
    }

    m_bCapturing = false;

    VisionResponse resp;
    resp.type = "video";
    resp.frameCount = static_cast<int>(frames.size());
    resp.width = frames.empty() ? 0 : frames[0].width;
    resp.height = frames.empty() ? 0 : frames[0].height;

    for (const auto& frame : frames)
    {
        resp.frames.push_back(Base64Encode(frame.jpegData));
    }

    jsonResponse = BuildJsonResponse(resp);
    DriverLog("CVisionServer: Captured %zu frames\n", frames.size());
    return true;
}

bool CVisionServer::HandleGetStatus(std::string& jsonResponse)
{
    VisionResponse resp;
    resp.type = "status";
    resp.message = m_bCapturing ? "capturing" : "ready";
    resp.width = 0;
    resp.height = 0;
    resp.frameCount = 0;
    jsonResponse = BuildJsonResponse(resp);
    return true;
}

bool CVisionServer::ProcessRequest(const std::string& jsonRequest, std::string& jsonResponse)
{
    if (!m_bInitialized)
    {
        VisionResponse resp;
        resp.type = "error";
        resp.message = "Vision server not initialized";
        resp.width = 0;
        resp.height = 0;
        resp.frameCount = 0;
        jsonResponse = BuildJsonResponse(resp);
        return false;
    }

    DriverLog("CVisionServer: Processing request: %s\n", jsonRequest.c_str());

    size_t actionPos = jsonRequest.find("\"action\"");
    if (actionPos == std::string::npos)
    {
        VisionResponse resp;
        resp.type = "error";
        resp.message = "Missing action field";
        resp.width = 0;
        resp.height = 0;
        resp.frameCount = 0;
        jsonResponse = BuildJsonResponse(resp);
        return false;
    }

    if (jsonRequest.find("capture_frame") != std::string::npos)
    {
        return HandleCaptureFrame(jsonResponse);
    }
    else if (jsonRequest.find("capture_video") != std::string::npos)
    {
        float duration = 3.0f;
        int fps = 10;

        size_t durPos = jsonRequest.find("\"duration\"");
        if (durPos != std::string::npos)
        {
            size_t colonPos = jsonRequest.find(':', durPos);
            if (colonPos != std::string::npos)
            {
                duration = static_cast<float>(std::atof(jsonRequest.c_str() + colonPos + 1));
            }
        }

        size_t fpsPos = jsonRequest.find("\"fps\"");
        if (fpsPos != std::string::npos)
        {
            size_t colonPos = jsonRequest.find(':', fpsPos);
            if (colonPos != std::string::npos)
            {
                fps = std::atoi(jsonRequest.c_str() + colonPos + 1);
            }
        }

        return HandleCaptureVideo(duration, fps, jsonResponse);
    }
    else if (jsonRequest.find("get_status") != std::string::npos)
    {
        return HandleGetStatus(jsonResponse);
    }

    VisionResponse resp;
    resp.type = "error";
    resp.message = "Unknown action";
    resp.width = 0;
    resp.height = 0;
    resp.frameCount = 0;
    jsonResponse = BuildJsonResponse(resp);
    return false;
}
