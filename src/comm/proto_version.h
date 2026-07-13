#pragma once
#include <cstdint>

// Heimdall wire-protocol version. Must match Java ProtoVersion.PROTO_VERSION and the
// proto_version field stamped on every top-level message (§2F). Bump on any incompatible
// change to a message's field semantics so a jar/image skew is detected, not mis-parsed.
namespace heimdall {
inline constexpr uint32_t PROTO_VERSION = 1;
}
