#include "app/heimdall_app.h"
#include "config/app_config_loader.h"
#include <csignal>
#include <cstdio>

static HeimdallApp* g_app = nullptr;

static void shutdown(int) {
    if (g_app) g_app->stop();
}

int main() {
    HeimdallApp app(load_app_config("config/heimdall.jsonc"));
    g_app = &app;
    std::signal(SIGINT, shutdown);
    std::printf("Heimdall starting. Ctrl+C to stop.\n");
    app.run();
    return 0;
}
