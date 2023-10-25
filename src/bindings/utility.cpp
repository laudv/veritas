#include "bindings.h"
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace veritas;

/** Convert between Python list and C++ Veritas box. */
Box::BufT tobox(py::object pybox) {
    Box::BufT buf;
    Box box{buf};
    FeatId count = 0;

    for (const auto& x : pybox) {
        Interval d;
        if (py::isinstance<py::tuple>(x)) {
            py::tuple t = py::cast<py::tuple>(x);
            FeatId id = py::cast<FeatId>(t[0]);
            d = py::cast<Interval>(t[1]);
            count = id;
        }
        else if (py::isinstance<py::int_>(x)) { // iterating dict
            FeatId id = py::cast<FeatId>(x);
            d = py::cast<Interval>(pybox[x]);
            count = id;
        }
        else if (py::isinstance<Interval>(x)) {
            d = py::cast<Interval>(x);
        }

        if (!box.refine_box(count, d))
            throw std::runtime_error("invalid box");

        //for (auto bb : box)//debug print
        //{
        //    if (bb.feat_id == count)
        //    {
        //        std::cout << "- in box:   " << bb.domain << " equal? "
        //            << (bb.domain.lo == d.lo) << (bb.domain.hi == d.hi) << std::endl
        //            << "  in pybox: " << d << std::endl;
        //    }
        //}
        ++count;
    }
    return buf;
}

data<FloatT> get_data(py::handle h, size_t min_num_cols) {
    auto arr = py::array::ensure(h);
    if (!arr) throw std::runtime_error("invalid eval array");
    if (!arr.dtype().is(pybind11::dtype::of<FloatT>()))
        throw std::runtime_error("invalid dtype");

    py::buffer_info buf = arr.request();
    data d { static_cast<FloatT *>(buf.ptr), 0, 0, 0, 0 };
    if (buf.ndim == 1) {
        d.num_rows = 1;
        d.num_cols = buf.shape[0];
        d.stride_row = 0; // there is only one row
        d.stride_col = buf.strides[0] / sizeof(FloatT);
    } else if (buf.ndim == 2) {
        d.num_rows = buf.shape[0];
        d.num_cols = buf.shape[1];
        d.stride_row = buf.strides[0] / sizeof(FloatT);
        d.stride_col = buf.strides[1] / sizeof(FloatT);
    }
    else throw py::value_error("invalid data");

    if (d.num_cols < min_num_cols)
        throw std::runtime_error("not enough columns in the data");

    return d;
}
