#pragma once

// Output of pose estimation (Milestone 2). Input to the tracker.
struct FieldDetection {
    int   class_id;
    float x, y;         // field-relative, meters (WPILib coordinate system)
    float confidence;
};
