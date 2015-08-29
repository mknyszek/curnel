# curnel

Curnel
------

Curnel provides a decorator to apply to functions such that they will be compiled on-the-fly and run in a CUDA context.
The actual runtime part, however, is provided by PyCUDA. Curnel provides instead a Python to CUDA C compiler, 
written in pure Python.

Features
--------

* Local variable definition with integer and float type inference
* Basic control flow (if, while)
* Full arithmetic support (including power operator and floor division)
* Clean, non-intrusive syntax
* Written in pure Python
* Full threadIdx, blockIdx, blockDim support
* Support for integer and float literals
* Support for indexing multiple levels deep
* Types are obtained from Python object; no type annotations!
* Support for constants (implemented as C macros)
* Support for Python syntactic sugar (tuples in indexing, inline if-else, etc.)

Usage
-----

Function definition:
```@cuda(height=2700, width=3600, range_scale=99./32.)
def mapping(cm, ss, rgbimg):
    x = threadIdx.x + blockIdx.x * blockDim.x
    y = threadIdx.y + blockIdx.y * blockDim.y
    row = height - x - 1
    if x < height and y < width:
        val = ss[x,y]
        if val > 0.:
            maploc = val * range_scale
            rgbimg[row,x,0] = cm[maploc,0]
            rgbimg[row,x,1] = cm[maploc,1]
            rgbimg[row,x,2] = cm[maploc,2]
        elif val <= -100.:
            rgbimg[row,x,0] = 32
            rgbimg[row,x,1] = 32
            rgbimg[row,x,2] = 32
        elif val > -2. and val <= 0.:
            d = -125. * val
            rgbimg[row,x,0] = 255
            rgbimg[row,x,1] = d
            rgbimg[row,x,2] = 255
        else:
            rgbimg[row,x,0] = 190
            rgbimg[row,x,1] = 190
            rgbimg[row,x,2] = 190```
  
Invocation:
  
```x = mapping(cm, ss, rgbimg)
print x```

Output:

```#include <stdint.h>

#define width 3600
#define range_scale 3.09375
#define height 2700

__global__ void mapping(uint8_t* cm, float* ss, uint8_t* rgbimg) {
    int32_t x = (threadIdx.x+(blockIdx.x*blockDim.x));
    int32_t y = (threadIdx.y+(blockIdx.y*blockDim.y));
    int32_t row = ((height-x)-1);
    if (((x < height) && (y < width))) {
        float val = ss[x * 3600 + y];
        if ((val > 0.0)) {
            float maploc = (val*range_scale);
            rgbimg[row * 3 * 3600 + x * 3 + 0] = cm[maploc * 3 + 0];
            rgbimg[row * 3 * 3600 + x * 3 + 1] = cm[maploc * 3 + 1];
            rgbimg[row * 3 * 3600 + x * 3 + 2] = cm[maploc * 3 + 2];
        } else if ((val <= -100.0)) {
            rgbimg[row * 3 * 3600 + x * 3 + 0] = 32;
            rgbimg[row * 3 * 3600 + x * 3 + 1] = 32;
            rgbimg[row * 3 * 3600 + x * 3 + 2] = 32;
        } else if (((val > -2.0) && (val <= 0.0))) {
            float d = (-125.0*val);
            rgbimg[row * 3 * 3600 + x * 3 + 0] = 255;
            rgbimg[row * 3 * 3600 + x * 3 + 1] = d;
            rgbimg[row * 3 * 3600 + x * 3 + 2] = 255;
        } else {
            rgbimg[row * 3 * 3600 + x * 3 + 0] = 190;
            rgbimg[row * 3 * 3600 + x * 3 + 1] = 190;
            rgbimg[row * 3 * 3600 + x * 3 + 2] = 190;
        }
        
    }
}```

TODO
----

* Integrate with PyCUDA (GPU memory management, run kernel)
* Support for loops with range()
* Shared memory syntax
* Complex number support
