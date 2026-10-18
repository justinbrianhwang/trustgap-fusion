from rrf import collector


def test_interfuser_rig_keeps_original_specs_and_saved_keys():
    expected_ids = ("rgb_front", "rgb_left", "rgb_right", "lidar", "imu")
    expected_dirs = ("rgb_front", "rgb_left", "rgb_right", "lidar", "measurements", "actors_data")
    assert tuple(spec["id"] for spec in collector.SENSOR_SPECS) == expected_ids
    assert collector.capture_subdirs("interfuser") == expected_dirs
    assert collector.build_parser().parse_args([]).rig == "interfuser"

