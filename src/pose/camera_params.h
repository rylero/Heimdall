#pragma once
#include <array>
#include <cmath>

struct CameraIntrinsics {
    float fx, fy;   // focal lengths, pixels
    float cx, cy;   // principal point, pixels
};

struct CameraExtrinsics {
    float tx, ty, tz;       // camera position in robot frame, meters (tz = height above ground)
    std::array<float, 9> R; // rotation matrix camera->robot, row-major 3x3
};

struct CameraParams {
    CameraIntrinsics intrinsics;
    CameraExtrinsics extrinsics;
};

// Robot position and heading in field frame (WPiLib standard coordinates).
struct RobotPose {
    float x, y;       // field-relative, meters
    float heading;    // radians, CCW positive from +X axis
};

// Build a camera->robot rotation matrix from mounting angles.
//
// Convention (camera frame = OpenCV: X right, Y down, Z forward):
//   yaw   -- which direction the camera faces on the robot, radians CCW from robot +X.
//            yaw=0 -> forward, yaw=pi/2 -> left side, yaw=-pi/2 -> right side.
//   pitch -- downward tilt of optical axis, radians. pitch=0 -> level, pitch>0 -> looking down.
//   roll  -- rotation around optical axis, radians (usually 0).
//
// Returns 9-element row-major 3x3 rotation matrix.
inline std::array<float, 9> rotation_from_euler(float yaw, float pitch, float roll) {
    // Base rotation: level, forward-facing camera (yaw=0, pitch=0, roll=0)
    //   cam_Z (forward) -> robot_X,  cam_X (right) -> -robot_Y,  cam_Y (down) -> -robot_Z
    const float R_base[9] = {
         0.f,  0.f, 1.f,
        -1.f,  0.f, 0.f,
         0.f, -1.f, 0.f,
    };

    // Rx(-pitch): pitch down rotates around cam_X toward cam_Y (down)
    const float cp = std::cos(-pitch), sp = std::sin(-pitch);
    const float Rx[9] = {
        1.f,  0.f,  0.f,
        0.f,  cp,  -sp,
        0.f,  sp,   cp,
    };

    // Rz(roll): roll around cam_Z
    const float cr = std::cos(roll), sr = std::sin(roll);
    const float Rz_cam[9] = {
         cr, -sr, 0.f,
         sr,  cr, 0.f,
        0.f, 0.f, 1.f,
    };

    // Rz(yaw): mount orientation around robot_Z
    const float cy2 = std::cos(yaw), sy2 = std::sin(yaw);
    const float Rz_rob[9] = {
         cy2, -sy2, 0.f,
         sy2,  cy2, 0.f,
        0.f,   0.f, 1.f,
    };

    // Helper: C = A * B (3x3 row-major)
    auto mat3mul = [](const float A[9], const float B[9], float C[9]) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                C[i*3+j] = 0.f;
                for (int k = 0; k < 3; ++k)
                    C[i*3+j] += A[i*3+k] * B[k*3+j];
            }
    };

    // R = Rz_rob * R_base * Rx(-pitch) * Rz_cam(roll)
    float tmp1[9], tmp2[9], result[9];
    mat3mul(R_base, Rx, tmp1);
    mat3mul(tmp1, Rz_cam, tmp2);
    mat3mul(Rz_rob, tmp2, result);

    std::array<float, 9> out;
    for (int i = 0; i < 9; ++i) out[i] = result[i];
    return out;
}
