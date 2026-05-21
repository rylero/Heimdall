package com.heimdall;

import java.util.Collections;
import java.util.List;

/** One detection frame received from the Jetson. */
public final class DetectionFrame {
    private final List<TrackEvent> events;
    private final long timestampNs;
    private final boolean healthy;

    public DetectionFrame(List<TrackEvent> events, long timestampNs, boolean healthy) {
        this.events = Collections.unmodifiableList(events);
        this.timestampNs = timestampNs;
        this.healthy = healthy;
    }

    /** Track lifecycle events (CONFIRMED / UPDATED / LOST) in this frame. */
    public List<TrackEvent> getEvents()    { return events; }
    public long getTimestampNs()           { return timestampNs; }
    /** False when the Jetson pipeline has an error (lost camera, inference crash, etc.). */
    public boolean isHealthy()             { return healthy; }
}
