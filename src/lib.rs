use numpy::ndarray::{ArrayViewD, Axis};
use numpy::PyArray2;
use opencv::imgcodecs::IMREAD_UNCHANGED;
use pyo3::prelude::*;
use opencv::prelude::*;
use crate::element_type::OpenCvElement;
use crate::mat_ext::MatExt;

fn main() -> anyhow::Result<()> {
    let img = opencv::imgcodecs::imread("a1.png", IMREAD_UNCHANGED).unwrap();
    let shape = img.size_with_depth();
    let array = ArrayViewD::from_shape(shape, &img.as_slice::<u8>()?)?;
    let mask = array.remove_axis(Axis(2));
    println!("{:?}", mask.shape());
    let mut standard_layout = mask.as_standard_layout();
    let slice = standard_layout.as_slice().unwrap();
    let shape_with_channels: Vec<i32> = mask.shape().iter().map(|&sz| sz as i32).collect();
    let (channels, shape) =  match shape_with_channels.split_last() {
        Some(split) => split,
        None => {
            Err(anyhow::anyhow!("err")).unwrap()
        }
    };


    println!("chan: {}", *channels);
    let mat = Mat::from_slice(slice).unwrap().reshape_nd(*channels, shape)?;


    Ok(())
}

mod element_type {
    use super::*;

    pub trait OpenCvElement {
        const DEPTH: i32;
    }

    impl OpenCvElement for u8 {
        const DEPTH: i32 = opencv::core::CV_8U;
    }

    impl OpenCvElement for i8 {
        const DEPTH: i32 = opencv::core::CV_8S;
    }

    impl OpenCvElement for u16 {
        const DEPTH: i32 = opencv::core::CV_16U;
    }

    impl OpenCvElement for i16 {
        const DEPTH: i32 = opencv::core::CV_16S;
    }

    impl OpenCvElement for i32 {
        const DEPTH: i32 = opencv::core::CV_32S;
    }

    impl OpenCvElement for f32 {
        const DEPTH: i32 = opencv::core::CV_32F;
    }

    impl OpenCvElement for f64 {
        const DEPTH: i32 = opencv::core::CV_64F;
    }
}
mod mat_ext {
    use std::slice;
    use anyhow::ensure;
    use crate::element_type::OpenCvElement;
    use super::*;

    pub trait MatExt {
        fn size_with_depth(&self) -> Vec<usize>;
        fn as_slice<T>(&self) -> anyhow::Result<&[T]>
        where
            T: OpenCvElement;

        fn numel(&self) -> usize {
            self.size_with_depth().iter().product()
        }

    }

    impl MatExt for opencv::core::Mat {
        fn size_with_depth(&self) -> Vec<usize> {
            let size = self.mat_size();
            let size = size.iter().map(|&dim| dim as usize);
            let channels = self.channels() as usize;
            size.chain([channels]).collect()
        }

        fn as_slice<T>(&self) -> anyhow::Result<&[T]>
        where
            T: OpenCvElement,
        {
            ensure!(self.depth() == T::DEPTH, "element type mismatch");
            ensure!(self.is_continuous(), "Mat data must be continuous");

            let numel = self.numel();
            let ptr = self.ptr(0)? as *const T;

            let slice = unsafe { slice::from_raw_parts(ptr, numel) };
            Ok(slice)
        }
    }
}
