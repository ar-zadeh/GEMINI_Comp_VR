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

// Windows Audio API for per-process mute/unmute (SteamVR sessions)
#include <mmdeviceapi.h>
#include <endpointvolume.h>
#include <audiopolicy.h>
#include <functiondiscoverykeys_devpkey.h>
#include <combaseapi.h>
#include <Psapi.h>
#include <algorithm>
#include <cctype>

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

    // Check if this is an audio command (mute/unmute)
    if (IsAudioCommand(message))
    {
        HandleAudioCommand(message);
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
        DriverLog("[PUSH] controller2 at %s - calling TrackedDevicePoseUpdated\n", recvTs.c_str());
        vr::VRServerDriverHost()->TrackedDevicePoseUpdated(
            g_pController2Driver->GetObjectId(),
            g_pController2Driver->GetPose(),
            sizeof(vr::DriverPose_t));
        DriverLog("[DONE] controller2 TrackedDevicePoseUpdated returned\n");
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
        
        // Debug: Log when any button is pressed
        if (pose.input.triggerClick || pose.input.buttonA || pose.input.buttonB || 
            pose.input.menu || pose.input.grip || pose.input.system)
        {
            DriverLog("[INPUT] Received: trigger=%d, A=%d, B=%d, menu=%d, grip=%d, system=%d, triggerValue=%.2f\n",
                pose.input.triggerClick, pose.input.buttonA, pose.input.buttonB,
                pose.input.menu, pose.input.grip, pose.input.system, pose.input.triggerValue);
        }
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

bool CPoseDataReceiver::IsAudioCommand(const std::string& json)
{
    size_t typePos = json.find("\"type\"");
    if (typePos == std::string::npos) return false;

    size_t colonPos = json.find(':', typePos);
    if (colonPos == std::string::npos) return false;

    size_t valuePos = json.find("\"audio_command\"", colonPos);
    return valuePos != std::string::npos;
}

// Helper: get lowercase executable name from a process ID
static std::string GetProcessName(DWORD pid)
{
    std::string name;
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (hProcess)
    {
        char buf[MAX_PATH];
        DWORD size = MAX_PATH;
        if (QueryFullProcessImageNameA(hProcess, 0, buf, &size))
        {
            // Extract filename from full path
            std::string fullPath(buf);
            size_t pos = fullPath.find_last_of("\\/");
            name = (pos != std::string::npos) ? fullPath.substr(pos + 1) : fullPath;
            // Lowercase
            std::transform(name.begin(), name.end(), name.begin(),
                [](unsigned char c) { return (char)std::tolower(c); });
        }
        CloseHandle(hProcess);
    }
    return name;
}

// Helper: check if a process name belongs to SteamVR or VR apps
// NOTE: processName is already lowercased by GetProcessName(), so all entries here must be lowercase
static bool IsSteamVRProcess(const std::string& processName)
{
    static const char* steamvrNames[] = {
        "vrserver.exe",
        "vrcompositor.exe",
        "vrdashboard.exe",
        "vrmonitor.exe",
        "vrwebhelper.exe",
        "steamvr_vrcompositor.exe",
        "vrstartup.exe",
        "steamwebhelper.exe",
        "vrchat.exe",
    };
    for (const char* name : steamvrNames)
    {
        if (processName == name)
            return true;
    }
    return false;
}

void CPoseDataReceiver::HandleAudioCommand(const std::string& json)
{
    DriverLog("CPoseDataReceiver: Handling audio command: %s\n", json.c_str());

    // Parse the action field: "mute", "unmute", "toggle", or "get_state"
    std::string action;
    size_t actionPos = json.find("\"action\"");
    if (actionPos != std::string::npos)
    {
        size_t colonPos = json.find(':', actionPos);
        if (colonPos != std::string::npos)
        {
            size_t valStart = json.find('"', colonPos + 1);
            if (valStart != std::string::npos)
            {
                valStart++;
                size_t valEnd = json.find('"', valStart);
                if (valEnd != std::string::npos)
                {
                    action = json.substr(valStart, valEnd - valStart);
                }
            }
        }
    }

    if (action.empty())
    {
        DriverLog("CPoseDataReceiver: Audio command missing action field\n");
        if (m_sendCallback)
        {
            m_sendCallback("{\"type\":\"audio_response\",\"success\":false,\"message\":\"Missing action field\"}\n");
        }
        return;
    }

    // Use Windows Core Audio Session API to mute only SteamVR processes
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    bool comInitialized = SUCCEEDED(hr) || hr == S_FALSE;

    IMMDeviceEnumerator* pEnumerator = nullptr;
    IMMDevice* pDevice = nullptr;
    IAudioSessionManager2* pSessionManager = nullptr;
    IAudioSessionEnumerator* pSessionEnum = nullptr;
    bool success = false;
    std::string message;
    bool newMuteState = false;
    int sessionsAffected = 0;

    do
    {
        hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_ALL,
            __uuidof(IMMDeviceEnumerator), (void**)&pEnumerator);
        if (FAILED(hr))
        {
            message = "Failed to create device enumerator";
            DriverLog("CPoseDataReceiver: %s (hr=0x%08lx)\n", message.c_str(), hr);
            break;
        }

        // Get the default audio render device (speakers/headphones — where SteamVR outputs)
        hr = pEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &pDevice);
        if (FAILED(hr))
        {
            message = "No default audio output device";
            DriverLog("CPoseDataReceiver: %s (hr=0x%08lx)\n", message.c_str(), hr);
            break;
        }

        // Get the session manager to enumerate per-process audio sessions
        hr = pDevice->Activate(__uuidof(IAudioSessionManager2), CLSCTX_ALL, NULL, (void**)&pSessionManager);
        if (FAILED(hr))
        {
            message = "Failed to activate session manager";
            DriverLog("CPoseDataReceiver: %s (hr=0x%08lx)\n", message.c_str(), hr);
            break;
        }

        hr = pSessionManager->GetSessionEnumerator(&pSessionEnum);
        if (FAILED(hr))
        {
            message = "Failed to get session enumerator";
            DriverLog("CPoseDataReceiver: %s (hr=0x%08lx)\n", message.c_str(), hr);
            break;
        }

        int sessionCount = 0;
        pSessionEnum->GetCount(&sessionCount);
        DriverLog("CPoseDataReceiver: Found %d audio sessions, scanning for SteamVR...\n", sessionCount);

        for (int i = 0; i < sessionCount; i++)
        {
            IAudioSessionControl* pSessionControl = nullptr;
            IAudioSessionControl2* pSessionControl2 = nullptr;
            ISimpleAudioVolume* pSimpleVolume = nullptr;

            hr = pSessionEnum->GetSession(i, &pSessionControl);
            if (FAILED(hr) || !pSessionControl) continue;

            hr = pSessionControl->QueryInterface(__uuidof(IAudioSessionControl2), (void**)&pSessionControl2);
            if (FAILED(hr) || !pSessionControl2)
            {
                pSessionControl->Release();
                continue;
            }

            // Get the process ID that owns this audio session
            DWORD pid = 0;
            pSessionControl2->GetProcessId(&pid);

            std::string procName = GetProcessName(pid);

            if (IsSteamVRProcess(procName))
            {
                DriverLog("CPoseDataReceiver: Found SteamVR session: %s (pid=%lu)\n", procName.c_str(), pid);

                hr = pSessionControl->QueryInterface(__uuidof(ISimpleAudioVolume), (void**)&pSimpleVolume);
                if (SUCCEEDED(hr) && pSimpleVolume)
                {
                    if (action == "mute")
                    {
                        pSimpleVolume->SetMute(TRUE, NULL);
                        newMuteState = true;
                        sessionsAffected++;
                    }
                    else if (action == "unmute")
                    {
                        pSimpleVolume->SetMute(FALSE, NULL);
                        newMuteState = false;
                        sessionsAffected++;
                    }
                    else if (action == "toggle")
                    {
                        BOOL currentMute = FALSE;
                        pSimpleVolume->GetMute(&currentMute);
                        newMuteState = !currentMute;
                        pSimpleVolume->SetMute(!currentMute, NULL);
                        sessionsAffected++;
                    }
                    else if (action == "get_state")
                    {
                        BOOL currentMute = FALSE;
                        pSimpleVolume->GetMute(&currentMute);
                        newMuteState = currentMute != FALSE;
                        sessionsAffected++;
                    }

                    pSimpleVolume->Release();
                }
            }

            pSessionControl2->Release();
            pSessionControl->Release();
        }

        if (sessionsAffected > 0)
        {
            success = true;
            char buf[128];
            sprintf_s(buf, sizeof(buf), "%s (%d SteamVR sessions)",
                newMuteState ? "muted" : "unmuted", sessionsAffected);
            message = buf;
            DriverLog("CPoseDataReceiver: SteamVR audio %s\n", message.c_str());
        }
        else
        {
            message = "No active SteamVR audio sessions found";
            DriverLog("CPoseDataReceiver: %s\n", message.c_str());
        }

    } while (false);

    // Cleanup COM objects
    if (pSessionEnum) pSessionEnum->Release();
    if (pSessionManager) pSessionManager->Release();
    if (pDevice) pDevice->Release();
    if (pEnumerator) pEnumerator->Release();
    if (comInitialized) CoUninitialize();

    // Send response back
    if (m_sendCallback)
    {
        char resp[512];
        sprintf_s(resp, sizeof(resp),
            "{\"type\":\"audio_response\",\"success\":%s,\"muted\":%s,\"target\":\"steamvr\",\"sessions\":%d,\"message\":\"%s\"}\n",
            success ? "true" : "false",
            newMuteState ? "true" : "false",
            sessionsAffected,
            message.c_str());
        m_sendCallback(std::string(resp));
    }
}
