part1_128_boat:
	python3 part1.py a 128 ../sample_image/boat.png ../sample_image/boat.png

part1_256_boat:
	python3 part1.py a 256 ../sample_image/boat.png ../sample_image/boat.png

part1_128_baboon:
	python3 part1.py a 128 ../sample_image/baboon.png ../sample_image/baboon.png

part1_256_baboon:
	python3 part1.py a 256 ../sample_image/baboon.png ../sample_image/baboon.png

part1_128_10imgs_boat:
	python3 part1.py b 128 ../sample_image/airplane.png ../sample_image/arctichare.png ../sample_image/baboon.png ../sample_image/barbara.png ../sample_image/sails.png ../sample_image/cameraman.tif ../sample_image/cat.png ../sample_image/fruits.png ../sample_image/peppers.png ../sample_image/mountain.png ../sample_image/boat.png

part1_256_10imgs_boat:
	python3 part1.py b 256 ../sample_image/airplane.png ../sample_image/arctichare.png ../sample_image/baboon.png ../sample_image/barbara.png ../sample_image/sails.png ../sample_image/cameraman.tif ../sample_image/cat.png ../sample_image/fruits.png ../sample_image/peppers.png ../sample_image/mountain.png ../sample_image/boat.png

part1_128_10imgs_sails:
	python3 part1.py b 128 ../sample_image/airplane.png ../sample_image/arctichare.png ../sample_image/baboon.png ../sample_image/barbara.png ../sample_image/boat.png ../sample_image/cameraman.tif ../sample_image/cat.png ../sample_image/fruits.png ../sample_image/peppers.png ../sample_image/mountain.png ../sample_image/sails.png

part1_256_10imgs_sails:
	python3 part1.py b 256 ../sample_image/airplane.png ../sample_image/arctichare.png ../sample_image/baboon.png ../sample_image/barbara.png ../sample_image/boat.png ../sample_image/cameraman.tif ../sample_image/cat.png ../sample_image/fruits.png ../sample_image/peppers.png ../sample_image/mountain.png ../sample_image/sails.png

part1_128_10imgs_baboon:
	python3 part1.py b 128 ../sample_image/airplane.png ../sample_image/arctichare.png ../sample_image/baboon.png ../sample_image/barbara.png ../sample_image/boat.png ../sample_image/cameraman.tif ../sample_image/cat.png ../sample_image/fruits.png ../sample_image/peppers.png ../sample_image/mountain.png ../sample_image/baboon.png

part1_256_10imgs_baboon:
	python3 part1.py b 256 ../sample_image/airplane.png ../sample_image/arctichare.png ../sample_image/baboon.png ../sample_image/barbara.png ../sample_image/boat.png ../sample_image/cameraman.tif ../sample_image/cat.png ../sample_image/fruits.png ../sample_image/peppers.png ../sample_image/mountain.png ../sample_image/baboon.png


# ========================================================================

part2a_boat:
	python3 part2.py a ../sample_image/boat.png

part2a_baboon:
	python3 part2.py a ../sample_image/baboon.png

part2a_airplane:
	python3 part2.py a ../sample_image/airplane.png

part2b_boat:
	python3 part2.py b ../sample_image/boat.png

part2b_baboon:
	python3 part2.py b ../sample_image/baboon.png

part2b_airplane:
	python3 part2.py b ../sample_image/airplane.png


clean:
	rm tmpFigs/*.png