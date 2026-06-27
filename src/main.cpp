#include "app/heimdall_app.h"
#include "config/app_config_loader.h"
#include <cstdlib>
#include <csignal>
#include <cstdio>
#include <optional>
#include <string>

static HeimdallApp* g_app = nullptr;

static void shutdown(int) {
    if (g_app) g_app->stop();
}

int main() {
    HeimdallApp::Config cfg = load_app_config("config/heimdall.jsonc");

    if (const char* rec = std::getenv("HEIMDALL_RECORD")) {
        cfg.record_path = std::string(rec);
        std::printf("[record] writing to %s\n", rec);
    }

    HeimdallApp app(std::move(cfg));
    g_app = &app;
    std::signal(SIGINT, shutdown);
    std::printf("Heimdall starting. Ctrl+C to stop.\n");
    app.run();
    return 0;
}
