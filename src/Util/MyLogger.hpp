#include <string>
#include <chrono>
#include <ctime>

class MyLogger {
public:
    enum class LogLevel {
        INFO,
        WARNING,
        ERROR
    };

    MyLogger() = default;

    static void Log(const std::string &message, LogLevel level = LogLevel::INFO) {
        auto now = std::chrono::system_clock::now();
        time_t now_time = std::chrono::system_clock::to_time_t(now);
        
        // 1. Thread-safe local time evaluation
        std::tm local_tm;
#ifdef _WIN32
        localtime_s(&local_tm, &now_time); // Windows / MSVC
#else
        localtime_r(&now_time, &local_tm); // POSIX / MinGW GCC
#endif

        // 2. Format the time string to "YYYY/MM/DD HH:MM:SS"
        char time_buf[32];
        std::strftime(time_buf, sizeof(time_buf), "%Y/%m/%d %H:%M:%S", &local_tm);

        // 3. Resolve the log level string
        const char* level_str = "INFO";
        switch (level) {
            case LogLevel::INFO:    level_str = "INFO"; break;
            case LogLevel::WARNING: level_str = "WARNING"; break;
            case LogLevel::ERROR:   level_str = "ERROR"; break;
        }

        // 4. Print everything in one clean line
        std::printf("[%s] %s %s\n", level_str, time_buf, message.c_str());
    }
};