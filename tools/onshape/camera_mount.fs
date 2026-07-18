FeatureScript 2278;
import(path : "onshape/std/geometry.fs", version : "2278.0");

// Camera Mount Frame
// ------------------
// Measures a camera's mount rotation for a Heimdall camera config by clicking the
// lens (front) face plus one "sensor-up" edge. Emits, into the FeatureScript notices,
// the camera->robot rotation matrix R (row-major, paste into extrinsics."R") and the
// equivalent yaw/pitch/roll in radians (paste into extrinsics."yaw"/"pitch"/"roll").
//
// Frames (must match src/pose/camera_params.h):
//   camera = OpenCV:  +Z out the lens, +X sensor-right, +Y sensor-down
//   robot  = WPILib:  +X forward,      +Y left,          +Z up
//
// The face normal fixes the optical axis (Z) -> yaw + pitch. The up-edge fixes the
// in-plane orientation -> roll. Do NOT use this to model a 90/180/270 sensor turn;
// that is the discrete "rotation" enum in the camera config, kept separate on purpose.

annotation { "Feature Type Name" : "Camera Mount Frame" }
export const cameraMountFrame = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Front (lens) face",
                     "Filter" : EntityType.FACE && GeometryType.PLANE,
                     "MaxNumberOfPicks" : 1 }
        definition.frontFace is Query;

        annotation { "Name" : "Sensor-up edge",
                     "Filter" : EntityType.EDGE && GeometryType.LINE,
                     "MaxNumberOfPicks" : 1 }
        definition.upEdge is Query;

        annotation { "Name" : "Robot origin (optional, defaults to world)",
                     "Filter" : BodyType.MATE_CONNECTOR,
                     "MaxNumberOfPicks" : 1 }
        definition.robotOrigin is Query;

        annotation { "Name" : "Flip optical axis (if normal points into the camera)" }
        definition.flipZ is boolean;
    }
    {
        // --- optical axis Z: outward normal of the lens face -----------------
        const plane = evFaceTangentPlane(context,
            { "face" : definition.frontFace, "parameter" : vector(0.5, 0.5) });
        var zAxis = normalize(plane.normal);
        if (definition.flipZ) zAxis = -zAxis;

        // --- sensor-up reference from the edge; OpenCV Y is sensor-DOWN ------
        const edge = evLine(context, { "edge" : definition.upEdge });
        var yAxis = -normalize(edge.direction);              // down = -up
        yAxis = normalize(yAxis - dot(yAxis, zAxis) * zAxis); // orthonormalize vs Z
        const xAxis = cross(yAxis, zAxis);                    // RH OpenCV: X = Y x Z, Z = X x Y

        // --- robot reference frame ------------------------------------------
        var rX; var rY; var rZ;
        if (isQueryEmpty(context, definition.robotOrigin))
        {
            rX = vector(1, 0, 0);   // WPILib world: X fwd, Y left, Z up
            rY = vector(0, 1, 0);
            rZ = vector(0, 0, 1);
        }
        else
        {
            const cs = evMateConnector(context, { "mateConnector" : definition.robotOrigin });
            rX = normalize(cs.xAxis);
            rZ = normalize(cs.zAxis);
            rY = cross(rZ, rX);
        }

        // express each camera axis in robot coordinates (columns of R)
        const Xr = vector(dot(xAxis, rX), dot(xAxis, rY), dot(xAxis, rZ));
        const Yr = vector(dot(yAxis, rX), dot(yAxis, rY), dot(yAxis, rZ));
        const Zr = vector(dot(zAxis, rX), dot(zAxis, rY), dot(zAxis, rZ));

        // R (camera->robot), row-major 3x3 = [ Xr | Yr | Zr ] as columns
        const R = [ Xr[0], Yr[0], Zr[0],
                    Xr[1], Yr[1], Zr[1],
                    Xr[2], Yr[2], Zr[2] ];

        // --- decompose to this codebase's roll->pitch->yaw convention -------
        // camera_params.h: R = Rz_rob(yaw) . R_base . Rx(-pitch) . Rz_cam(roll)
        //   Z_cam in robot = (cosP cosY, cosP sinY, -sinP)
        const yaw   = atan2(Zr[1], Zr[0]);
        const pitch = asin(-Zr[2]);
        // roll: compare measured X to the roll=0 reference axes in the sensor plane
        const cy = cos(yaw); const sy = sin(yaw);
        const X0 = vector(sy, -cy, 0 * unitless);
        const Y0 = vector(-sin(pitch) * cy, -sin(pitch) * sy, -cos(pitch));
        const roll = atan2(dot(Xr, Y0), dot(Xr, X0));

        // --- surface the results --------------------------------------------
        // radians (what config wants) + degrees for a sanity read
        println("=== Camera Mount Frame ===");
        println("R (row-major, paste into extrinsics.\"R\"):");
        println("  [ " ~ toString(R[0]) ~ ", " ~ toString(R[1]) ~ ", " ~ toString(R[2]) ~ ",");
        println("    " ~ toString(R[3]) ~ ", " ~ toString(R[4]) ~ ", " ~ toString(R[5]) ~ ",");
        println("    " ~ toString(R[6]) ~ ", " ~ toString(R[7]) ~ ", " ~ toString(R[8]) ~ " ]");
        println("yaw   = " ~ toString(yaw   / radian) ~ " rad (" ~ toString(yaw   / degree) ~ " deg)");
        println("pitch = " ~ toString(pitch / radian) ~ " rad (" ~ toString(pitch / degree) ~ " deg)");
        println("roll  = " ~ toString(roll  / radian) ~ " rad (" ~ toString(roll  / degree) ~ " deg)");

        // Drop a mate connector at the camera optical frame as a visual check, and
        // name it with the angles so the values are visible in the Part Studio tree.
        opMateConnector(context, id + "cameraMC", {
                "coordSystem" : coordSystem(plane.origin, xAxis, zAxis),
                "owner"       : qOwnerBody(definition.frontFace)
        });
        setProperty(context, {
                "entities" : qCreatedBy(id + "cameraMC", EntityType.BODY),
                "propertyType" : PropertyType.NAME,
                "value" : "cam y=" ~ toString(round(yaw / degree)) ~
                          " p=" ~ toString(round(pitch / degree)) ~
                          " r=" ~ toString(round(roll / degree))
        });
    });
