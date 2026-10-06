from main import HEAD_REGION, TORSO_REGION, wears

person = (100, 100, 200, 400)  # 100 wide, 300 tall
hat_on_head = (120, 90, 180, 150)
hat_on_floor = (300, 380, 360, 420)
vest_on_torso = (105, 180, 195, 300)

assert wears(person, [hat_on_head], HEAD_REGION)
assert not wears(person, [hat_on_floor], HEAD_REGION)
assert not wears(person, [], HEAD_REGION)
assert wears(person, [vest_on_torso], TORSO_REGION)
assert not wears(person, [hat_on_head], TORSO_REGION)  # helmet is not a vest region hit
print("ok")
