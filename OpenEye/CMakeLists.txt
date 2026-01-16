include_directories(../../headers)

set(TARGET_NAME driver_null)

add_library(${TARGET_NAME} MODULE
  basics.cpp
  basics.h
  cframecapture.cpp
  cframecapture.h
  cposedatareceiver.cpp
  cposedatareceiver.h
  ctcpclient.cpp
  ctcpclient.h
  cvisionserver.cpp
  cvisionserver.h
  driverlog.cpp
  driverlog.h
  driver_sample.cpp
  csampledevicedriver.cpp
  csampledevicedriver.h
  cserverdriver_sample.cpp
  cserverdriver_sample.h
  csamplecontrollerdriver.cpp
  csamplecontrollerdriver.h
  cwatchdogdriver_sample.cpp
  cwatchdogdriver_sample.h
  jpge.cpp
  jpge.h
)

# --- DEPLOYMENT CONFIGURATION ---
# Force output to a structure compatible with SteamVR driver layout
# This ensures that a simple Build (without Install) creates a valid driver folder.

if(CMAKE_SIZEOF_VOID_P STREQUAL "8")
    set(ARCH 64)
else()
    set(ARCH 32) # Just in case
endif()

# Determine binary output path, e.g., build/drivers/sample/bin/win64
set(DRIVER_ROOT "${CMAKE_BINARY_DIR}/drivers/sample")
set(DRIVER_BIN_DIR "${DRIVER_ROOT}/bin/${PREFIX}${ARCH}")

set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${DRIVER_BIN_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${DRIVER_BIN_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${DRIVER_BIN_DIR}"
)

# Copy resources to the build tree structure so the driver works immediately after build
add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DRIVER_ROOT}/resources/input"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DRIVER_ROOT}/resources/settings"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/mycontroller_profile.json"
        "${DRIVER_ROOT}/resources/input/mycontroller_profile.json"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/default.vrsettings"
        "${DRIVER_ROOT}/resources/settings/default.vrsettings"
    COMMENT "Copying driver resources to build tree structure"
)

install(TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION drivers/sample/bin/${PREFIX}${ARCH}
    LIBRARY DESTINATION drivers/sample/bin/${PREFIX}${ARCH}
)

install(FILES default.vrsettings DESTINATION drivers/sample/resources/settings)
install(FILES mycontroller_profile.json DESTINATION drivers/sample/resources/input)
