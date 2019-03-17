all:
	swig3.0 -c++ -python MLX90640.i
	python3 setup.py build