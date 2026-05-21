package com.heimdall;

public final class TrackEvent {
    private final TrackEventType type;
    private final TrackedObject object;

    public TrackEvent(TrackEventType type, TrackedObject object) {
        this.type = type;
        this.object = object;
    }

    public TrackEventType getType()    { return type; }
    public TrackedObject  getObject()  { return object; }

    @Override
    public String toString() {
        return type + " " + object;
    }
}
