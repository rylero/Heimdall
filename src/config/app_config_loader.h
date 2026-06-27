#pragma once
#include "app/heimdall_app.h"
#include <string>

// Loads config/heimdall.json → HeimdallApp::Config.
// Camera intrinsics/extrinsics are loaded from the cameras_dir listed in that file.
HeimdallApp::Config load_app_config(const std::string& path);
