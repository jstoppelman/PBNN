Ensure you have the following packages

```
gau2grid: conda install -c psi4 gau2grid
pint: conda install -c psi4 pint
pydantic: conda install -c conda-forge pydantic
pybind11: conda install -c psi4 pybind11
libxc: conda install -c psi4 libxc
libint2: conda install -c psi4/label/dev libint2
qcengine: conda install -c psi4 qcengine
qcelemental: conda install -c psi4 qcelemental
```

Next, 
```
git clone "https://github.com/johnppederson/psi4"
cd psi4
mkdir build
cmake -S. -Bbuild -DCMAKE_INSTALL_PREFIX=/your_prefix/psi4/psi4/bin
```

Then make
```
cd build
make -j`getconf _NPROCESSORS_ONLN`
make install
```
