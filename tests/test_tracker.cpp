#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "tracker/tracker.h"

static std::vector<FieldDetection> at(float x, float y, int cls = 0) {
    return {{cls, x, y, 0.9f}};
}

TEST_CASE("new detection produces no CONFIRMED event until confirmation_frames met", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 3});
    auto events = t.update(at(1.f, 2.f), 0.0);
    bool any_confirmed = std::any_of(events.begin(), events.end(),
        [](const TrackEvent& e){ return e.type == TrackEventType::CONFIRMED; });
    REQUIRE_FALSE(any_confirmed);
}

TEST_CASE("track confirmed exactly at confirmation_frames", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 3});
    std::vector<TrackEvent> events;
    for (int i = 0; i < 3; ++i)
        events = t.update(at(1.f, 2.f), i * 0.02);
    long confirmed = std::count_if(events.begin(), events.end(),
        [](const TrackEvent& e){ return e.type == TrackEventType::CONFIRMED; });
    REQUIRE(confirmed == 1);
}

TEST_CASE("confirmed track emits UPDATED every subsequent frame", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 2});
    for (int i = 0; i < 2; ++i)
        t.update(at(0.f, 0.f), i * 0.02);
    auto events = t.update(at(0.f, 0.f), 2 * 0.02);
    long updated = std::count_if(events.begin(), events.end(),
        [](const TrackEvent& e){ return e.type == TrackEventType::UPDATED; });
    REQUIRE(updated == 1);
}

TEST_CASE("confirmed track emits LOST after loss_frames missed frames", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 2, .loss_frames = 3, .gate_distance = 0.1f});
    for (int i = 0; i < 2; ++i)
        t.update(at(0.f, 0.f), i * 0.02);
    std::vector<TrackEvent> events;
    for (int i = 0; i < 3; ++i)
        events = t.update({}, (2 + i) * 0.02);
    long lost = std::count_if(events.begin(), events.end(),
        [](const TrackEvent& e){ return e.type == TrackEventType::LOST; });
    REQUIRE(lost == 1);
}

TEST_CASE("after LOST, no further events for that track", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 1, .loss_frames = 1, .gate_distance = 0.1f});
    t.update(at(0.f, 0.f), 0.0);
    t.update({}, 0.02);
    auto events = t.update({}, 0.04);
    bool any = std::any_of(events.begin(), events.end(),
        [](const TrackEvent& e){
            return e.type == TrackEventType::LOST || e.type == TrackEventType::UPDATED;
        });
    REQUIRE_FALSE(any);
}

TEST_CASE("two well-separated objects tracked with correct class IDs", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 2, .gate_distance = 0.5f});
    std::vector<FieldDetection> two = {{0, 0.f, 0.f, 0.9f}, {1, 10.f, 10.f, 0.9f}};
    std::vector<TrackEvent> events;
    for (int i = 0; i < 2; ++i)
        events = t.update(two, i * 0.02);
    long confirmed = std::count_if(events.begin(), events.end(),
        [](const TrackEvent& e){ return e.type == TrackEventType::CONFIRMED; });
    REQUIRE(confirmed == 2);
    std::vector<int> class_ids;
    for (const auto& e : events)
        if (e.type == TrackEventType::CONFIRMED)
            class_ids.push_back(e.object.class_id);
    std::sort(class_ids.begin(), class_ids.end());
    REQUIRE(class_ids == std::vector<int>{0, 1});
}

TEST_CASE("tentative track discarded on first missed frame", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 5, .gate_distance = 0.1f});
    t.update(at(0.f, 0.f), 0.0);
    t.update({}, 0.02);
    for (int i = 0; i < 5; ++i)
        t.update(at(0.f, 0.f), (2 + i) * 0.02);
    auto events = t.update(at(0.f, 0.f), 7 * 0.02);
    long confirmed = std::count_if(events.begin(), events.end(),
        [](const TrackEvent& e){ return e.type == TrackEventType::CONFIRMED; });
    REQUIRE(confirmed == 0);  // discarded track can't confirm; fresh track confirmed in prior frame
}

TEST_CASE("CONFIRMED event carries correct position", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 2});
    std::vector<TrackEvent> events;
    for (int i = 0; i < 2; ++i)
        events = t.update(at(3.f, 7.f), i * 0.02);
    auto it = std::find_if(events.begin(), events.end(),
        [](const TrackEvent& e){ return e.type == TrackEventType::CONFIRMED; });
    REQUIRE(it != events.end());
    REQUIRE_THAT(it->object.x, Catch::Matchers::WithinAbs(3.f, 0.5f));
    REQUIRE_THAT(it->object.y, Catch::Matchers::WithinAbs(7.f, 0.5f));
}
