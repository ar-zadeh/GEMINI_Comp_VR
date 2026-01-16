#include "cposedatareceiver.h"
#include "cvisionserver.h"
#include "cserverdriver_sample.h"
#include "driverlog.h"
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>
#include <atomic>
#include <chrono>

CPoseDataReceiver* g_pPoseDataReceiver = nullptr;

CPoseDataReceiver::CPoseDataReceiver()
    : m_bMonitorRunning(false)
{
}

CPoseDataReceiver::~CPoseDataReceiver()
{
    Stop();
}

bool CPoseDataReceiver::Start(const std::string& host, int port)
{
    m_host = host;
    m_port = port;

    m_tcpClient.SetMessageCallback([this](const std::string& msg) {
        OnMessageReceived(msg);
    });

    m_bMonitorRunning = true;
    m_monitorThread = std::thread(&CPoseDataReceiver::MonitorConnectionThread, this);
    
    DriverLog("CPoseDataReceiver: Started connection monitor for %s:%d\n", host.c_str(), port);
    return true;
}

void CPoseDataReceiver::Stop()
{
    m_bMonitorRunning = false;
    if (m_monitorThread.joinable())
    {
        m_monitorThread.join();
    }

    m_tcpClient.StopReceiveThread();
    m_tcpClient.Disconnect();
}

void CPoseDataReceiver::MonitorConnectionThread()
{
    while (m_bMonitorRunning)
    {
        if (!m_tcpClient.IsConnected())
        {
            // Ensure previous thread is cleaned up
             m_tcpClient.StopReceiveThread();
             
             // Try to connect
             if (m_tcpClient.Connect(m_host, m_port))
             {
                 m_tcpClient.StartReceiveThread();
                 DriverLog("CPoseDataReceiver: Connection established\n");
             }
             else 
             {
                 // Wait before retry
                 std::this_thread::sleep_for(std::chrono::seconds(2));
             }
        }
        else
        {
             std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

bool CPoseDataReceiver::IsConnected() const
{
    return m_tcpClient.IsConnected();
}

PoseData CPoseDataReceiver::GetHeadsetPose()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    PoseData pose = m_headsetPose;
    m_headsetPose.updated = false;
    return pose;
}

PoseData CPoseDataReceiver::GetController1Pose()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    PoseData pose = m_controller1Pose;
    m_controller1Pose.updated = false;
    return pose;
}

PoseData CPoseDataReceiver::GetController2Pose()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    PoseData pose = m_controller2Pose;
    m_controller2Pose.updated = false;
    return pose;
}

void CPoseDataReceiver::OnMessageReceived(const std::string& message)
{
    // Check if this is a vision request
    if (IsVisionRequest(message))
    {
        HandleVisionRequest(message);
        return;
    }

    std::string device;
    PoseData pose;

    if (!ParseJson(message, device, pose))
    {
        DriverLog("CPoseDataReceiver: Failed to parse message: %s\n", message.c_str());
        return;
    }

    // Get receive timestamp for controller2 latency debugging
    std::string recvTs;
    if (device == "controller2")
    {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        std::time_t time = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
#if defined(_WIN32)
        localtime_s(&tm, &time);
#else
        tm = *std::localtime(&time);
#endif
        char buf[32];
#if defined(_WIN32)
        sprintf_s(buf, sizeof(buf), "%02d:%02d:%02d.%03d", tm.tm_hour, tm.tm_min, tm.tm_sec, (int)ms.count());
#else
        sprintf(buf, "%02d:%02d:%02d.%03d", tm.tm_hour, tm.tm_min, tm.tm_sec, (int)ms.count());
#endif
        recvTs = buf;
        
        // Extract send_ts from message and log comparison
        size_t tsPos = message.find("\"send_ts\":\"");
        if (tsPos != std::string::npos)
        {
            std::string sendTs = message.substr(tsPos + 11, 12);
            DriverLog("[RECV] controller2 | sent=%s | recv=%s\n", sendTs.c_str(), recvTs.c_str());
        }
        else
        {
            DriverLog("[RECV] controller2 at %s (no send_ts in msg)\n", recvTs.c_str());
        }
    }

    // Store pose data
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (device == "headset")
        {
            m_headsetPose = pose;
            m_headsetPose.updated = true;
        }
        else if (device == "controller1")
        {
            m_controller1Pose = pose;
            m_controller1Pose.updated = true;
        }
        else if (device == "controller2")
        {
            m_controller2Pose = pose;
            m_controller2Pose.updated = true;
        }
    } // Release mutex before calling OpenVR

    // PUSH-BASED UPDATE: Immediately notify SteamVR of pose change
    // This bypasses the slow RunFrame() polling and reduces latency from ~4s to near-instant
    if (device == "headset" && g_pHeadsetDriver && g_pHeadsetDriver->IsActivated())
    {
        vr::VRServerDriverHost()->TrackedDevicePoseUpdated(
            g_pHeadsetDriver->GetObjectId(),
            g_pHeadsetDriver->GetPose(),
            sizeof(vr::DriverPose_t));
    }
    else if (device == "controller1" && g_pController1Driver && g_pController1Driver->IsActivated())
    {
        vr::VRServerDriverHost()->TrackedDevicePoseUpdated(
            g_pController1Driver->GetObjectId(),
            g_pController1Driver->GetPose(),
            sizeof(vr::DriverPose_t));
    }
    else if (device == "controller2" && g_pController2Driver && g_pController2Driver->IsActivated())
    {
        DriverLog("[PUSH] controller2 at %s\n", recvTs.c_str());
        vr::VRServerDriverHost()->TrackedDevicePoseUpdated(
            g_pController2Driver->GetObjectId(),
            g_pController2Driver->GetPose(),
            sizeof(vr::DriverPose_t));
    }
}

// Simple JSON parser for our specific format:
// {"device":"headset","pos":[0.0,1.5,0.0],"rot":[0.0,0.0,0.0]}
// Or with input: {"device":"controller1","pos":[...],"rot":[...],"input":{...}}
bool CPoseDataReceiver::ParseJson(const std::string& json, std::string& device, PoseData& pose)
{
    // Find device
    size_t deviceStart = json.find("\"device\"");
    if (deviceStart == std::string::npos) return false;

    size_t colonPos = json.find(':', deviceStart);
    if (colonPos == std::string::npos) return false;

    size_t valueStart = json.find('"', colonPos);
    if (valueStart == std::string::npos) return false;
    valueStart++;

    size_t valueEnd = json.find('"', valueStart);
    if (valueEnd == std::string::npos) return false;

    device = json.substr(valueStart, valueEnd - valueStart);

    // Find pos array
    size_t posStart = json.find("\"pos\"");
    if (posStart == std::string::npos) return false;

    size_t bracketStart = json.find('[', posStart);
    if (bracketStart == std::string::npos) return false;

    size_t bracketEnd = json.find(']', bracketStart);
    if (bracketEnd == std::string::npos) return false;

    std::string posStr = json.substr(bracketStart + 1, bracketEnd - bracketStart - 1);

    // Parse pos values
    double posValues[3] = {0, 0, 0};
    int posIndex = 0;
    size_t start = 0;
    size_t comma;
    while (posIndex < 3 && (comma = posStr.find(',', start)) != std::string::npos)
    {
        posValues[posIndex++] = std::atof(posStr.substr(start, comma - start).c_str());
        start = comma + 1;
    }
    if (posIndex < 3)
    {
        posValues[posIndex] = std::atof(posStr.substr(start).c_str());
    }

    pose.posX = posValues[0];
    pose.posY = posValues[1];
    pose.posZ = posValues[2];

    // Find rot array
    size_t rotStart = json.find("\"rot\"");
    if (rotStart == std::string::npos) return false;

    bracketStart = json.find('[', rotStart);
    if (bracketStart == std::string::npos) return false;

    bracketEnd = json.find(']', bracketStart);
    if (bracketEnd == std::string::npos) return false;

    std::string rotStr = json.substr(bracketStart + 1, bracketEnd - bracketStart - 1);

    // Parse rot values
    double rotValues[3] = {0, 0, 0};
    int rotIndex = 0;
    start = 0;
    while (rotIndex < 3 && (comma = rotStr.find(',', start)) != std::string::npos)
    {
        rotValues[rotIndex++] = std::atof(rotStr.substr(start, comma - start).c_str());
        start = comma + 1;
    }
    if (rotIndex < 3)
    {
        rotValues[rotIndex] = std::atof(rotStr.substr(start).c_str());
    }

    pose.rotX = rotValues[0];
    pose.rotY = rotValues[1];
    pose.rotZ = rotValues[2];

    // Parse input object if present (for controllers)
    size_t inputStart = json.find("\"input\"");
    if (inputStart != std::string::npos)
    {
        pose.input.inputUpdated = true;
        
        // Parse boolean buttons
        pose.input.system = json.find("\"system\":true", inputStart) != std::string::npos;
        pose.input.menu = json.find("\"menu\":true", inputStart) != std::string::npos;
        pose.input.grip = json.find("\"grip\":true", inputStart) != std::string::npos;
        pose.input.triggerClick = json.find("\"triggerClick\":true", inputStart) != std::string::npos;
        pose.input.trackpadClick = json.find("\"trackpadClick\":true", inputStart) != std::string::npos;
        pose.input.trackpadTouch = json.find("\"trackpadTouch\":true", inputStart) != std::string::npos;
        pose.input.buttonA = json.find("\"buttonA\":true", inputStart) != std::string::npos;
        pose.input.buttonB = json.find("\"buttonB\":true", inputStart) != std::string::npos;

        // Parse analog values
        auto parseFloat = [&json, inputStart](const char* key, float defaultVal) -> float {
            std::string searchKey = std::string("\"") + key + "\":";
            size_t keyPos = json.find(searchKey, inputStart);
            if (keyPos == std::string::npos) return defaultVal;
            size_t valStart = keyPos + searchKey.length();
            // Skip whitespace
            while (valStart < json.length() && (json[valStart] == ' ' || json[valStart] == '\t')) valStart++;
            size_t valEnd = valStart;
            while (valEnd < json.length() && (isdigit(json[valEnd]) || json[valEnd] == '.' || json[valEnd] == '-')) valEnd++;
            if (valEnd > valStart) {
                return static_cast<float>(std::atof(json.substr(valStart, valEnd - valStart).c_str()));
            }
            return defaultVal;
        };

        pose.input.triggerValue = parseFloat("triggerValue", 0.0f);
        pose.input.joystickX = parseFloat("joystickX", 0.0f);
        pose.input.joystickY = parseFloat("joystickY", 0.0f);
    }

    return true;
}

bool CPoseDataReceiver::IsVisionRequest(const std::string& json)
{
    // Check for "type" field with "vision_request" value (handle optional spaces after colon)
    size_t typePos = json.find("\"type\"");
    if (typePos == std::string::npos) return false;
    
    size_t colonPos = json.find(':', typePos);
    if (colonPos == std::string::npos) return false;
    
    // Look for "vision_request" after the colon
    size_t valuePos = json.find("\"vision_request\"", colonPos);
    return valuePos != std::string::npos;
}

void CPoseDataReceiver::HandleVisionRequest(const std::string& json)
{
    DriverLog("CPoseDataReceiver: Handling vision request: %s\n", json.c_str());

    if (!g_pVisionServer)
    {
        DriverLog("CPoseDataReceiver: ERROR - Vision server is NULL!\n");
        if (m_sendCallback)
        {
            std::string errorResp = "{\"type\":\"error\",\"message\":\"Vision server not initialized\",\"width\":0,\"height\":0,\"frameCount\":0}\n";
            m_sendCallback(errorResp);
            DriverLog("CPoseDataReceiver: Sent error response\n");
        }
        else
        {
            DriverLog("CPoseDataReceiver: ERROR - No send callback set!\n");
        }
        return;
    }

    std::string response;
    bool success = g_pVisionServer->ProcessRequest(json, response);
    DriverLog("CPoseDataReceiver: Vision request processed, success=%d, response size=%zu\n", 
              success, response.size());

    if (m_sendCallback)
    {
        m_sendCallback(response + "\n");
        DriverLog("CPoseDataReceiver: Vision response sent\n");
    }
    else
    {
        DriverLog("CPoseDataReceiver: ERROR - No send callback, cannot send response!\n");
    }
}
