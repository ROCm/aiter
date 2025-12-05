#ifndef TOY_FORMAT_HPP
#define TOY_FORMAT_HPP

#include <string>
#include <sstream>
#include <utility>

namespace std {

namespace detail {

template <class T>
std::string to_string_helper(const T& v) {
    std::ostringstream os;
    os << v;
    return os.str();
}

inline std::string format_impl(std::string_view fmt) {
    return std::string(fmt);
}

template <class Arg, class... Args>
std::string format_impl(std::string_view fmt,
                        const Arg& first,
                        const Args&... rest) {
    std::size_t pos = fmt.find("{}");
    if (pos == std::string_view::npos)
        throw std::runtime_error("extra argument provided to format");

    return std::string(fmt.substr(0, pos)) +
           to_string_helper(first) +
           format_impl(fmt.substr(pos + 2), rest...);
}

} // namespace detail

template <class... Args>
std::string format(std::string_view fmt, const Args&... args) {
    std::string result = detail::format_impl(fmt, args...);
    if (result.find("{}") != std::string::npos)
        throw std::runtime_error("too few arguments provided to format");
    return result;
}

} // namespace std

#endif // TOY_FORMAT_HPP
