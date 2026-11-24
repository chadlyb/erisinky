//! Zero-copy GGUF reader: mmaps the file and parses metadata + the tensor table.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::io::AsRawFd;

pub const T_F32: u32 = 0;
pub const T_Q5_0: u32 = 6;
pub const T_Q8_0: u32 = 8;
pub const T_Q4_K: u32 = 12;
pub const T_Q6_K: u32 = 14;
pub const T_BF16: u32 = 30;
/// inkserve's private expert types (NATIVE.md): NVFP4 and MXFP8 blocks of 32 weights
pub const T_NV4: u32 = 200;
pub const T_MX8: u32 = 201;

/// (bytes per block, elements per block)
pub fn type_block(t: u32) -> (usize, usize) {
    match t {
        0 => (4, 1),
        1 | 30 => (2, 1),
        2 => (18, 32),
        6 => (22, 32),
        8 => (34, 32),
        12 => (144, 256),
        13 => (176, 256),
        14 => (210, 256),
        200 => (18, 32),
        201 => (34, 32),
        _ => panic!("unsupported ggml type {t}"),
    }
}

pub fn type_name(t: u32) -> &'static str {
    match t {
        0 => "F32", 1 => "F16", 30 => "BF16", 2 => "Q4_0", 6 => "Q5_0", 8 => "Q8_0",
        12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K", 200 => "NV4", 201 => "MX8", _ => "?",
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U(u64),
    I(i64),
    F(f64),
    B(bool),
    S(String),
    AI(Vec<i64>),
    #[allow(dead_code)]
    AF(Vec<f64>),
    AS(Vec<String>),
    AB(Vec<bool>),
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub ne: Vec<u64>,
    pub ty: u32,
    /// byte offset from the start of the mapped file
    pub file_off: u64,
    pub nbytes: u64,
}

impl Tensor {
    pub fn row_bytes(&self) -> u64 {
        let (bs, be) = type_block(self.ty);
        self.ne[0] / be as u64 * bs as u64
    }
}

pub struct Gguf {
    _file: File,
    pub map: *mut u8,
    pub map_len: usize,
    pub file_len: usize,
    pub kv: HashMap<String, Value>,
    pub tensors: HashMap<String, Tensor>,
}

unsafe impl Send for Gguf {}
unsafe impl Sync for Gguf {}

impl Drop for Gguf {
    fn drop(&mut self) { unsafe { libc::munmap(self.map as *mut _, self.map_len); } }
}

struct Rd<'a> { b: &'a [u8], p: usize }
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] { let s = &self.b[self.p..self.p + n]; self.p += n; s }
    fn u8(&mut self) -> u8 { self.take(1)[0] }
    fn u16(&mut self) -> u16 { u16::from_le_bytes(self.take(2).try_into().unwrap()) }
    fn u32(&mut self) -> u32 { u32::from_le_bytes(self.take(4).try_into().unwrap()) }
    fn u64(&mut self) -> u64 { u64::from_le_bytes(self.take(8).try_into().unwrap()) }
    fn str(&mut self) -> String { let n = self.u64() as usize; String::from_utf8_lossy(self.take(n)).into_owned() }
    fn scalar(&mut self, t: u32) -> Value {
        match t {
            0 => Value::U(self.u8() as u64),
            1 => Value::I(self.u8() as i8 as i64),
            2 => Value::U(self.u16() as u64),
            3 => Value::I(self.u16() as i16 as i64),
            4 => Value::U(self.u32() as u64),
            5 => Value::I(self.u32() as i32 as i64),
            6 => Value::F(f32::from_bits(self.u32()) as f64),
            7 => Value::B(self.u8() != 0),
            8 => Value::S(self.str()),
            10 => Value::U(self.u64()),
            11 => Value::I(self.u64() as i64),
            12 => Value::F(f64::from_bits(self.u64())),
            9 => {
                let et = self.u32();
                let n = self.u64() as usize;
                match et {
                    8 => Value::AS((0..n).map(|_| self.str()).collect()),
                    7 => Value::AB((0..n).map(|_| self.u8() != 0).collect()),
                    6 | 12 => Value::AF((0..n).map(|_| match self.scalar(et) { Value::F(f) => f, _ => 0.0 }).collect()),
                    _ => Value::AI((0..n).map(|_| match self.scalar(et) { Value::U(u) => u as i64, Value::I(i) => i, _ => 0 }).collect()),
                }
            }
            _ => panic!("bad gguf value type {t}"),
        }
    }
}

impl Gguf {
    pub fn open(path: &str) -> std::io::Result<Gguf> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len() as usize;
        let page = 16384usize;
        let map_len = file_len.div_ceil(page) * page;
        let map = unsafe {
            libc::mmap(std::ptr::null_mut(), map_len, libc::PROT_READ, libc::MAP_SHARED, file.as_raw_fd(), 0)
        };
        if map == libc::MAP_FAILED { return Err(std::io::Error::last_os_error()); }
        let map = map as *mut u8;
        let bytes = unsafe { std::slice::from_raw_parts(map, file_len) };
        let mut r = Rd { b: bytes, p: 0 };
        assert_eq!(r.take(4), b"GGUF", "not a GGUF file");
        let _ver = r.u32();
        let nt = r.u64();
        let nkv = r.u64();
        let mut kv = HashMap::new();
        for _ in 0..nkv {
            let k = r.str();
            let t = r.u32();
            let v = r.scalar(t);
            kv.insert(k, v);
        }
        let align = match kv.get("general.alignment") { Some(Value::U(a)) => *a, _ => 32 };
        let mut tl = Vec::new();
        for _ in 0..nt {
            let name = r.str();
            let nd = r.u32();
            let ne: Vec<u64> = (0..nd).map(|_| r.u64()).collect();
            let ty = r.u32();
            let off = r.u64();
            let (bs, be) = type_block(ty);
            let n: u64 = ne.iter().product();
            tl.push(Tensor { name, ne, ty, file_off: off, nbytes: n / be as u64 * bs as u64 });
        }
        let data_off = (r.p as u64).div_ceil(align) * align;
        let mut tensors = HashMap::new();
        for mut t in tl {
            t.file_off += data_off;
            assert!(t.file_off + t.nbytes <= file_len as u64, "tensor {} out of bounds", t.name);
            tensors.insert(t.name.clone(), t);
        }
        Ok(Gguf { _file: file, map, map_len, file_len, kv, tensors })
    }

    pub fn u(&self, k: &str) -> u64 {
        match self.kv.get(k) {
            Some(Value::U(u)) => *u,
            Some(Value::I(i)) => *i as u64,
            Some(Value::B(b)) => *b as u64,
            v => panic!("kv {k} not an integer: {v:?}"),
        }
    }
    pub fn f(&self, k: &str) -> f64 {
        match self.kv.get(k) { Some(Value::F(f)) => *f, _ => self.u(k) as f64 }
    }
    pub fn tensor(&self, name: &str) -> &Tensor {
        self.tensors.get(name).unwrap_or_else(|| panic!("missing tensor {name}"))
    }
    pub fn opt(&self, name: &str) -> Option<&Tensor> { self.tensors.get(name) }
    pub fn bytes(&self, t: &Tensor) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.map.add(t.file_off as usize), t.nbytes as usize) }
    }
}
