# Imitation_learning
 본 Repo는 항공기 충돌회피 보조시스템 학습을 위한 policy net의 구조를 실험하는 내용이다.

## Environment(Data) design (datagen.py)
본체 시야 내부에 있는 항공기를 회피하는 것을 전제로 했기에, 상대기의 시작점은 본체의 이동방향에 대해 부채꼴 형식으로 +-50도 위치에서 출발하게 설정하였다.
본체는 NED 좌표계 기준 NE 평면의 (0, 0)에서 +N 방향으로 200의 속력으로 진행하고, 상대기는 
