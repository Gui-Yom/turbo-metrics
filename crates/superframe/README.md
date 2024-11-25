# superframe

Flexible and extensible typesafe structures for heterogeneous computing on image frames.

## Features

### Planes and Images, packed and/or planar channels

```rust
fn main() {
    // Planes are a 2D array of samples.
    // Single packed plane
    let plane = Plane::<Box<[u8; 3]>>::new(1024, 1024)?;
    // Images are a collection of planes.
    // Planar RGB image
    let plane = Image::<Box<[f32]>, Array<3>>::new(1024, 1024)?;
}
```

### Image metadata

```rust
fn main() {
    // Images can store user defined metadata
    // Here, Colormodel is a user defined type
    let img = Image::<Box<[u8]>, Array<3>, Colormodel>::new(1024, 1024, Colormodel::RGB)?;
    dbg!(img.metadata());
}
```

### Frames on the heap or on another device (with transfers)

```rust
fn main() {
    // Planes and images are generic over their storage.
    // Storage can be accessible from the host (cpu) or not.

    // Single u8 plane in gpu memory
    let mut gpu = Plane::<Cuda<u8>>::new(1024, 1024)?;
    // With explicit stream for allocation, default Drop impl will use the default stream by default.
    let plane2 = Plane::<Cuda<u8>>::new_ext(1024, 1024, &CuStream::DEFAULT)?;

    let cpu = Plane::<Box<[u8]>>::new(1024, 1024)?;
    cpu.set(255)?;

    // Generic transfers !
    gpu.copy_from(&cpu)?;
}
```

### Images with planes of different sizes and sample types

```rust
fn main() {
    let luma = Plane::<Cuda<u8>>::new(1024, 1024)?;
    let uv = Plane::<Cuda<[u8; 2]>>::new(512, 512)?;
    let nv12: Image<Cuda<DynSample>, Array<2>> = Image::from_planes([luma.to_dyn(), uv.to_dyn()]);
}
```

### Plane slicing

```rust
fn main() {
    let p = Plane::<Box<[u8]>>::new(1024, 1024)?;
    // This will only write in the top left 16x16 patch.
    p.view_mut(0, 0, 16, 16).set(127);
}
```

### Traits for generic programming

```rust
/// This function accepts anything that looks like a plane stored in cpu memory.
/// This includes views.
fn a(p: impl AsPlane<Storage: HostAccessible>) {}
/// This function accepts planes with any sample type whatever the storage is.
fn b<Stor: SampleStorage<SampleType=DynSample>>(p: Plane<Stor>) {
    dispatch_sample!(p;
        u8 => { println!("It's u8") }
        @other => { println!("It's something else") });
}

fn main() {
    let ext = Plane::<Box<[u8]>>::new(1024, 1024)?;
    a(&ext);
    a(ext.view(&ext.rect()));
    b(ext.to_dyn());
}
```

### Extensible

```rust
pub trait Sample: Copy {
    /// This is a tag stored out of band for DynSample to identify the real type at runtime
    type Id;
    const SIZE: usize = size_of::<Self>();
}

pub trait Device {
    type Storage<S>;

    /// Alloc a plane with this allocator, memory is not cleared and is set with whatever rubbish was here before.
    fn alloc<S: StaticSample>(
        &self,
        width: usize,
        height: usize,
    ) -> Result<(usize, Self::Storage<S>), Box<dyn Error>>;
    fn drop<S: Sample>(&self, stor: Self::Storage<S>) -> Result<(), Box<dyn Error>>;

    fn convert_to_dyn<S: StaticSample>(stor: Self::Storage<S>) -> Self::Storage<DynSample>;
    fn convert_from_dyn<S: StaticSample>(stor: Self::Storage<DynSample>) -> Self::Storage<S>;
}
```

### Import frames from outside

Frames from ffmpeg libraries or other APIs
