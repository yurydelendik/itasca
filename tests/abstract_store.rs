//! Executes Wasm against a toy embedder store whose handles are raw pointers
//! to embedder-owned objects, as an engine embedding itasca would use.
use itasca::runtime::*;
use itasca::{ConstExpr, ExportDesc, FuncType, ImportDesc, ValidationProfile};
use std::ptr::NonNull;

struct FuncObj {
    ty: FuncType,
    type_id: usize,
    callee: Callee<InstH, u64>,
}
struct TableObj {
    refs: Vec<u64>,
    max: Option<u32>,
}
struct MemObj {
    bytes: Vec<u8>,
    max: Option<u32>,
}
struct GlobalObj {
    bits: u128,
}
struct InstanceObj {
    module: ExecutableModule,
    funcs: Vec<FuncH>,
    tables: Vec<TableH>,
    memories: Vec<MemH>,
    globals: Vec<GlobalH>,
    types: Vec<usize>,
    elems: Vec<Vec<u64>>,
    datas: Vec<Vec<u8>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct FuncH(NonNull<FuncObj>);
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct TableH(NonNull<TableObj>);
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct MemH(NonNull<MemObj>);
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct GlobalH(NonNull<GlobalObj>);
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct InstH(NonNull<InstanceObj>);
/// A segment of an instance, addressed by the instance and its segment index.
#[derive(Clone, Copy)]
struct SegH(InstH, u32);
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ExternH(u64);

fn alloc<T>(owner: &mut Vec<*mut T>, value: T) -> NonNull<T> {
    let p = Box::into_raw(Box::new(value));
    owner.push(p);
    NonNull::new(p).unwrap()
}
fn free<T>(owner: &mut Vec<*mut T>) {
    for p in owner.drain(..) {
        drop(unsafe { Box::from_raw(p) });
    }
}

/// Owns every object; handles stay valid until the store is dropped.
#[derive(Default)]
struct PtrStore {
    funcs: Vec<*mut FuncObj>,
    tables: Vec<*mut TableObj>,
    memories: Vec<*mut MemObj>,
    globals: Vec<*mut GlobalObj>,
    instances: Vec<*mut InstanceObj>,
    signatures: Vec<FuncType>,
}
impl Drop for PtrStore {
    fn drop(&mut self) {
        free(&mut self.funcs);
        free(&mut self.tables);
        free(&mut self.memories);
        free(&mut self.globals);
        free(&mut self.instances);
    }
}
impl PtrStore {
    fn intern(&mut self, ty: &FuncType) -> usize {
        match self.signatures.iter().position(|t| t == ty) {
            Some(i) => i,
            None => {
                self.signatures.push(ty.clone());
                self.signatures.len() - 1
            }
        }
    }
    fn host(&mut self, ty: FuncType, id: u64) -> FuncH {
        let type_id = self.intern(&ty);
        FuncH(alloc(
            &mut self.funcs,
            FuncObj {
                ty,
                type_id,
                callee: Callee::Host(id),
            },
        ))
    }
    /// Instantiates a module that imports only functions and has no active
    /// segments or non-constant global initializers.
    fn instantiate(&mut self, module: ExecutableModule, imports: &[FuncH]) -> InstH {
        let env = module.env();
        let types: Vec<usize> = env.types().iter().map(|t| self.intern(t)).collect();
        let defined: Vec<u32> = env.func_type_indices().to_vec();
        let mut func_imports = 0;
        for import in env.imports() {
            assert!(matches!(import.desc, ImportDesc::Func(_)));
            func_imports += 1;
        }
        assert_eq!(func_imports, imports.len());
        let tables: Vec<TableH> = env
            .table_types()
            .iter()
            .map(|t| {
                TableH(alloc(
                    &mut self.tables,
                    TableObj {
                        refs: vec![0; t.limits.min as usize],
                        max: t.limits.max,
                    },
                ))
            })
            .collect();
        let memories: Vec<MemH> = env
            .mem_types()
            .iter()
            .map(|m| {
                MemH(alloc(
                    &mut self.memories,
                    MemObj {
                        bytes: vec![0; m.limits.min as usize * 65536],
                        max: m.limits.max,
                    },
                ))
            })
            .collect();
        let globals: Vec<GlobalH> = env
            .globals()
            .iter()
            .map(|g| {
                let bits = match g.init {
                    ConstExpr::I32(v) => v as u32 as u128,
                    ConstExpr::I64(v) => v as u64 as u128,
                    ConstExpr::F32(v) => v as u128,
                    ConstExpr::F64(v) => v as u128,
                    _ => panic!("unsupported initializer"),
                };
                GlobalH(alloc(&mut self.globals, GlobalObj { bits }))
            })
            .collect();
        let elems = vec![Vec::new(); env.elements().len()];
        let datas = module.tail().data.iter().map(|d| d.init.clone()).collect();
        let inst = InstH(alloc(
            &mut self.instances,
            InstanceObj {
                module,
                funcs: imports.to_vec(),
                tables,
                memories,
                globals,
                types: types.clone(),
                elems,
                datas,
            },
        ));
        for (i, &t) in defined.iter().enumerate() {
            let ty = self.signatures[types[t as usize]].clone();
            let f = FuncH(alloc(
                &mut self.funcs,
                FuncObj {
                    ty,
                    type_id: types[t as usize],
                    callee: Callee::Wasm {
                        instance: inst,
                        function: i,
                    },
                },
            ));
            unsafe { (&mut *inst.0.as_ptr()).funcs.push(f) };
        }
        inst
    }
    fn export(&self, inst: InstH, name: &str) -> FuncH {
        let i = unsafe { inst.0.as_ref() };
        for e in i.module.env().exports() {
            if e.name == name.as_bytes() {
                if let ExportDesc::Func(n) = e.desc {
                    return i.funcs[n as usize];
                }
            }
        }
        panic!("no export {name}")
    }
}

fn inst(i: InstH) -> &'static InstanceObj {
    unsafe { &*i.0.as_ptr() }
}

impl AbstractStore for PtrStore {
    type Instance = InstH;
    type Func = FuncH;
    type Table = TableH;
    type Memory = MemH;
    type Global = GlobalH;
    type Elem = SegH;
    type Data = SegH;
    type Extern = ExternH;
    type Host = u64;
    type TypeId = usize;

    fn identity(&self) -> usize {
        self as *const Self as usize
    }
    fn max_frames(&self) -> usize {
        1000
    }
    fn max_values(&self) -> usize {
        100_000
    }
    fn module(&self, i: InstH) -> &ExecutableModule {
        &inst(i).module
    }
    fn instance_func(&self, i: InstH, index: u32) -> FuncH {
        inst(i).funcs[index as usize]
    }
    fn instance_table(&self, i: InstH, index: u32) -> TableH {
        inst(i).tables[index as usize]
    }
    fn instance_has_memory(&self, i: InstH) -> bool {
        !inst(i).memories.is_empty()
    }
    fn instance_memory(&self, i: InstH, index: u32) -> MemH {
        inst(i).memories[index as usize]
    }
    fn instance_global(&self, i: InstH, index: u32) -> GlobalH {
        inst(i).globals[index as usize]
    }
    fn instance_elem(&self, i: InstH, index: u32) -> SegH {
        SegH(i, index)
    }
    fn instance_data(&self, i: InstH, index: u32) -> SegH {
        SegH(i, index)
    }
    fn instance_type_id(&self, i: InstH, index: u32) -> usize {
        inst(i).types[index as usize]
    }
    fn callee(&self, f: FuncH) -> Callee<InstH, u64> {
        unsafe { f.0.as_ref() }.callee
    }
    fn signature(&self, f: FuncH) -> &FuncType {
        &unsafe { &*f.0.as_ptr() }.ty
    }
    fn func_type_id(&self, f: FuncH) -> usize {
        unsafe { f.0.as_ref() }.type_id
    }
    fn owns_func(&self, f: FuncH) -> bool {
        self.funcs.contains(&f.0.as_ptr())
    }
    fn owns_extern(&self, _: ExternH) -> bool {
        true
    }
    fn func_to_ref(&self, f: FuncH) -> u64 {
        f.0.as_ptr() as usize as u64
    }
    fn func_from_ref(&self, r: u64) -> FuncH {
        FuncH(unsafe { NonNull::new_unchecked(r as usize as *mut FuncObj) })
    }
    fn extern_to_ref(&self, r: ExternH) -> u64 {
        r.0
    }
    fn extern_from_ref(&self, r: u64) -> ExternH {
        ExternH(r)
    }
    fn memory_bytes(&self, m: MemH) -> &[u8] {
        &unsafe { &*m.0.as_ptr() }.bytes
    }
    fn memory_bytes_mut(&mut self, m: MemH) -> &mut [u8] {
        &mut unsafe { &mut *m.0.as_ptr() }.bytes
    }
    fn memory_grow(&mut self, m: MemH, delta: u32) -> u32 {
        let m = unsafe { &mut *m.0.as_ptr() };
        let old = m.bytes.len() / 65536;
        let new = old as u64 + delta as u64;
        if new > m.max.unwrap_or(65536).min(65536) as u64 {
            return u32::MAX;
        }
        m.bytes.resize(new as usize * 65536, 0);
        old as u32
    }
    fn memory_and_data(&mut self, m: MemH, d: SegH) -> (&mut [u8], &[u8]) {
        let bytes = &mut unsafe { &mut *m.0.as_ptr() }.bytes;
        (bytes, &inst(d.0).datas[d.1 as usize])
    }
    fn data_drop(&mut self, d: SegH) {
        unsafe { (&mut *d.0 .0.as_ptr()).datas[d.1 as usize] = Vec::new() };
    }
    fn global_get(&self, g: GlobalH) -> u64 {
        unsafe { g.0.as_ref() }.bits as u64
    }
    fn global_get_v128(&self, g: GlobalH) -> u128 {
        unsafe { g.0.as_ref() }.bits
    }
    fn global_set(&mut self, g: GlobalH, value: u64) {
        unsafe { (*g.0.as_ptr()).bits = value as u128 };
    }
    fn global_set_v128(&mut self, g: GlobalH, value: u128) {
        unsafe { (*g.0.as_ptr()).bits = value };
    }
    fn table_len(&self, t: TableH) -> u32 {
        unsafe { t.0.as_ref() }.refs.len() as u32
    }
    fn table_ref(&self, t: TableH, index: u32) -> u64 {
        unsafe { t.0.as_ref() }.refs[index as usize]
    }
    fn set_table_ref(&mut self, t: TableH, index: u32, r: u64) {
        unsafe { (&mut *t.0.as_ptr()).refs[index as usize] = r };
    }
    fn table_grow(&mut self, t: TableH, delta: u32, init: u64) -> u32 {
        let t = unsafe { &mut *t.0.as_ptr() };
        let old = t.refs.len();
        let new = old as u64 + delta as u64;
        if new > t.max.unwrap_or(u32::MAX) as u64 {
            return u32::MAX;
        }
        t.refs.resize(new as usize, init);
        old as u32
    }
    fn elem_len(&self, e: SegH) -> u32 {
        inst(e.0).elems[e.1 as usize].len() as u32
    }
    fn elem_ref(&self, e: SegH, index: u32) -> u64 {
        inst(e.0).elems[e.1 as usize][index as usize]
    }
    fn elem_drop(&mut self, e: SegH) {
        unsafe { (&mut *e.0 .0.as_ptr()).elems[e.1 as usize] = Vec::new() };
    }
}

/// Host function 1 re-enters Wasm to call `inc` and multiplies its result by ten.
struct Embedder {
    inc: Option<FuncH>,
    host_calls: usize,
}
impl StorePlatform<PtrStore> for Embedder {
    fn interrupt(&mut self) -> InterruptAction {
        InterruptAction::Continue
    }
    fn call_host(
        &mut self,
        store: &mut PtrStore,
        host: u64,
        args: &[StoreValue<PtrStore>],
        results: &mut [StoreValue<PtrStore>],
    ) -> Result<(), HostError> {
        assert_eq!(host, 1);
        self.host_calls += 1;
        let mut out = Vec::new();
        let inc = self.inc.unwrap();
        StoreExecution::<PtrStore>::new()
            .invoke_in(store, self, inc, args, &mut out)
            .map_err(|_| HostError::Trap)?;
        match out[0] {
            Val::I32(v) => results[0] = Val::I32(v * 10),
            _ => return Err(HostError::Trap),
        }
        Ok(())
    }
}

const WAT: &str = r#"
(module
  (import "env" "host" (func $host (param i32) (result i32)))
  (type $t (func (param i32) (result i32)))
  (memory 1 3)
  (global $g (mut i32) (i32.const 0))
  (table 2 funcref)
  (elem declare func $inc $host)
  (func $inc (export "inc") (param i32) (result i32)
    (i32.add (local.get 0) (i32.const 1)))
  (func (export "init")
    (table.set (i32.const 0) (ref.func $inc))
    (table.set (i32.const 1) (ref.func $host)))
  (func (export "run") (param i32) (result i32)
    (i32.store (i32.const 0) (call_indirect (type $t) (local.get 0) (i32.const 0)))
    (global.set $g (i32.add (global.get $g) (i32.load (i32.const 0))))
    (call_indirect (type $t) (global.get $g) (i32.const 1)))
  (func (export "first") (result funcref) (table.get (i32.const 0)))
  (func (export "oob") (result i32)
    (call_indirect (type $t) (i32.const 0) (i32.const 5)))
  (func (export "grow") (result i32) (memory.grow (i32.const 1))))
"#;

#[test]
fn executes_against_pointer_handle_store() {
    let buf = wast::parser::ParseBuffer::new(WAT).unwrap();
    let mut ast = wast::parser::parse::<wast::Wat>(&buf).unwrap();
    let module = compile(&ast.encode().unwrap(), ValidationProfile::Wasm2).unwrap();

    let mut store = PtrStore::default();
    let host_ty = FuncType {
        params: vec![itasca::ValueType::I32],
        results: vec![itasca::ValueType::I32],
    };
    let host = store.host(host_ty, 1);
    let instance = store.instantiate(module, &[host]);
    let mut embedder = Embedder {
        inc: Some(store.export(instance, "inc")),
        host_calls: 0,
    };
    let mut exec = StoreExecution::<PtrStore>::new();
    let mut out = Vec::new();
    let mut call = |store: &mut PtrStore, name: &str, args: &[StoreValue<PtrStore>]| {
        let f = store.export(instance, name);
        exec.invoke_in(store, &mut embedder, f, args, &mut out)
            .map(|()| out.clone())
    };

    call(&mut store, "init", &[]).unwrap();
    // inc(5) = 6 is stored, g = 6, host(6) = inc(6) * 10.
    assert_eq!(
        call(&mut store, "run", &[Val::I32(5)]).unwrap(),
        [Val::I32(70)]
    );
    assert_eq!(
        call(&mut store, "run", &[Val::I32(5)]).unwrap(),
        [Val::I32(130)]
    );
    assert_eq!(
        &store.memory_bytes(inst(instance).memories[0])[..4],
        &6u32.to_le_bytes()
    );
    assert_eq!(store.global_get(inst(instance).globals[0]), 12);

    // A funcref result is the embedder's own function pointer.
    let inc = store.export(instance, "inc");
    assert_eq!(
        call(&mut store, "first", &[]).unwrap(),
        [Val::FuncRef(Some(inc))]
    );
    assert_eq!(
        call(&mut store, "oob", &[]),
        Err(ExecError::Trap(Trap::TableOutOfBounds))
    );
    assert_eq!(call(&mut store, "grow", &[]).unwrap(), [Val::I32(1)]);
    assert_eq!(call(&mut store, "grow", &[]).unwrap(), [Val::I32(2)]);
    assert_eq!(
        call(&mut store, "grow", &[]).unwrap(),
        [Val::I32(-1i32 as u32)]
    );
    drop(call);
    assert_eq!(embedder.host_calls, 2);
}
