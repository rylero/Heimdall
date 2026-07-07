package com.heimdall.proto;

/**
 * Heimdall wire-protocol version. Must match the C++ PROTO_VERSION (proto_version.h) and the
 * proto_version field stamped on every top-level message (§2F). Bump on any incompatible change
 * to a message's field semantics so a jar/image skew is detected instead of silently mis-parsed.
 */
public final class ProtoVersion {
    private ProtoVersion() {}

    public static final int PROTO_VERSION = 1;
}
