//! Reviewed native instantiations of the generic runtime.
use super::platform::Alloc;
use super::store::FuncCode;
use super::*;
use crate::{GlobalType, MemType, TableType};
use core::sync::atomic::{AtomicUsize, Ordering};

mod float;
pub(super) use float::NativeFloat;

pub(crate) struct StdAlloc;
impl Alloc for StdAlloc {
    fn reserve<T>(values: &mut Vec<T>, additional: usize) -> bool {
        values.try_reserve(additional).is_ok()
    }
}
static NEXT_STORE: AtomicUsize = AtomicUsize::new(1);
fn next_identity() -> usize {
    let mut id = NEXT_STORE.load(Ordering::Relaxed);
    loop {
        let next = id.checked_add(1).expect("store identity space exhausted");
        match NEXT_STORE.compare_exchange_weak(id, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return id,
            Err(now) => id = now,
        }
    }
}
impl MemoryBuffer for VecMemory {
    fn new_zeroed(len: usize) -> Option<Self> {
        Self::new_zeroed_with::<StdAlloc>(len)
    }
    fn bytes(&self) -> &[u8] {
        self.as_bytes()
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        self.as_bytes_mut()
    }
    fn grow(&mut self, new_len: usize) -> bool {
        self.grow_with::<StdAlloc>(new_len)
    }
}
impl<M: MemoryBuffer> Default for Store<M> {
    fn default() -> Self {
        Self::new()
    }
}
impl<M: MemoryBuffer> Store<M> {
    /// Creates an empty store with default resource limits.
    pub fn new() -> Self {
        Self::with_limits(StoreLimits::default())
    }
    /// Creates an empty store with explicit resource limits.
    pub fn with_limits(limits: StoreLimits) -> Self {
        Self::with_identity(next_identity(), limits)
    }
    /// Defines a synchronous host function with its declared type.
    pub fn define_host(&mut self, ty: FuncType, id: u64) -> Result<FuncAddr, ExecError> {
        self.define_host_with::<StdAlloc>(ty, id)
    }
    /// Defines an embedding-owned reference identity.
    pub fn define_extern(&mut self, id: u64) -> Result<ExternAddr, ExecError> {
        self.define_extern_with::<StdAlloc>(id)
    }
    /// Defines memory backed by exactly its declared initial number of pages.
    pub fn define_memory(&mut self, ty: MemType, buf: M) -> Result<MemoryAddr, ExecError> {
        self.define_memory_with::<StdAlloc>(ty, buf)
    }
    /// Defines a table filled with null references.
    pub fn define_table(&mut self, ty: TableType) -> Result<TableAddr, ExecError> {
        self.define_table_with::<StdAlloc>(ty)
    }
    /// Defines a global with a typed, store-owned initial value.
    pub fn define_global(&mut self, ty: GlobalType, value: Value) -> Result<GlobalAddr, ExecError> {
        self.define_global_with::<StdAlloc>(ty, value)
    }
    /// Links and initializes segments, returning the start function for the
    /// caller to invoke.
    ///
    /// Link errors and Wasm 1.0 segment bounds errors leave the store unchanged.
    /// Wasm 2.0 traps preserve completed segment writes and allocated objects.
    pub fn instantiate<P: Platform<M>>(
        &mut self,
        module: ExecutableModule,
        imports: &[Extern],
        platform: &mut P,
    ) -> Result<Instantiated, ExecError> {
        self.instantiate_with::<P, NativeFloat, StdAlloc>(module, imports, platform)
    }
    /// Grows a memory, returning its previous page count or `u32::MAX` on failure.
    pub fn grow_memory(&mut self, addr: MemoryAddr, delta: u32) -> Result<u32, ExecError> {
        self.grow_memory_with(addr, delta)
    }
    /// Grows a table, returning its previous length or `u32::MAX` on failure.
    pub fn grow_table(
        &mut self,
        addr: TableAddr,
        delta: u32,
        value: Value,
    ) -> Result<u32, ExecError> {
        self.grow_table_with::<StdAlloc>(addr, delta, value)
    }
}
impl<M: MemoryBuffer> AbstractStore for Store<M> {
    type Instance = usize;
    type Func = FuncAddr;
    type Table = usize;
    type Memory = usize;
    type Global = usize;
    type Elem = usize;
    type Data = usize;
    type Extern = ExternAddr;
    type Host = u64;
    type TypeId = usize;

    #[inline(always)]
    fn identity(&self) -> usize {
        self.id
    }
    #[inline(always)]
    fn max_frames(&self) -> usize {
        self.limits.max_frames
    }
    #[inline(always)]
    fn max_values(&self) -> usize {
        self.limits.max_values
    }
    #[inline(always)]
    fn module(&self, instance: usize) -> &ExecutableModule {
        &self.instances[instance].module
    }
    #[inline(always)]
    fn instance_func(&self, instance: usize, index: u32) -> FuncAddr {
        self.instances[instance].funcs[index as usize]
    }
    #[inline(always)]
    fn instance_table(&self, instance: usize, index: u32) -> usize {
        self.instances[instance].tables[index as usize].index
    }
    #[inline(always)]
    fn instance_has_memory(&self, instance: usize) -> bool {
        self.instances[instance].memories.len() > 0
    }
    #[inline(always)]
    fn instance_memory(&self, instance: usize, index: u32) -> usize {
        self.instances[instance].memories[index as usize].index
    }
    #[inline(always)]
    fn instance_global(&self, instance: usize, index: u32) -> usize {
        self.instances[instance].globals[index as usize].index
    }
    #[inline(always)]
    fn instance_elem(&self, instance: usize, index: u32) -> usize {
        self.instances[instance].elems[index as usize]
    }
    #[inline(always)]
    fn instance_data(&self, instance: usize, index: u32) -> usize {
        self.instances[instance].datas[index as usize]
    }
    #[inline(always)]
    fn instance_type_id(&self, instance: usize, index: u32) -> usize {
        self.instances[instance].types[index as usize]
    }
    #[inline(always)]
    fn callee(&self, func: FuncAddr) -> Callee<usize, u64> {
        match self.funcs[func.index].code {
            FuncCode::Wasm { instance, function } => Callee::Wasm { instance, function },
            FuncCode::Host(id) => Callee::Host(id),
        }
    }
    #[inline(always)]
    fn signature(&self, func: FuncAddr) -> &FuncType {
        &self.types[self.funcs[func.index].signature]
    }
    #[inline(always)]
    fn func_type_id(&self, func: FuncAddr) -> usize {
        self.funcs[func.index].signature
    }
    #[inline(always)]
    fn owns_func(&self, func: FuncAddr) -> bool {
        func.store == self.id
    }
    #[inline(always)]
    fn owns_extern(&self, r: ExternAddr) -> bool {
        r.store == self.id
    }
    #[inline(always)]
    fn func_to_ref(&self, func: FuncAddr) -> u64 {
        func.index as u64 + 1
    }
    #[inline(always)]
    fn func_from_ref(&self, r: u64) -> FuncAddr {
        FuncAddr {
            store: self.id,
            index: r as usize - 1,
        }
    }
    #[inline(always)]
    fn extern_to_ref(&self, r: ExternAddr) -> u64 {
        r.index as u64 + 1
    }
    #[inline(always)]
    fn extern_from_ref(&self, r: u64) -> ExternAddr {
        ExternAddr {
            store: self.id,
            index: r as usize - 1,
        }
    }
    #[inline(always)]
    fn memory_bytes(&self, memory: usize) -> &[u8] {
        self.memories[memory].buf.bytes()
    }
    #[inline(always)]
    fn memory_bytes_mut(&mut self, memory: usize) -> &mut [u8] {
        self.memories[memory].buf.bytes_mut()
    }
    #[inline(always)]
    fn memory_grow(&mut self, memory: usize, delta: u32) -> u32 {
        self.grow_memory_index(memory, delta)
    }
    #[inline(always)]
    fn memory_and_data(&mut self, memory: usize, data: usize) -> (&mut [u8], &[u8]) {
        (
            self.memories[memory].buf.bytes_mut(),
            &self.datas[data].bytes,
        )
    }
    #[inline(always)]
    fn data_drop(&mut self, data: usize) {
        self.datas[data].bytes = Vec::new();
    }
    #[inline(always)]
    fn global_get(&self, global: usize) -> u64 {
        self.globals[global].lo
    }
    #[inline(always)]
    fn global_get_v128(&self, global: usize) -> u128 {
        let g = &self.globals[global];
        g.lo as u128 | ((g.hi as u128) << 64)
    }
    #[inline(always)]
    fn global_set(&mut self, global: usize, value: u64) {
        self.globals[global].lo = value;
    }
    #[inline(always)]
    fn global_set_v128(&mut self, global: usize, value: u128) {
        let g = &mut self.globals[global];
        g.lo = value as u64;
        g.hi = (value >> 64) as u64;
    }
    #[inline(always)]
    fn table_len(&self, table: usize) -> u32 {
        self.tables[table].elements.len() as u32
    }
    #[inline(always)]
    fn table_ref(&self, table: usize, index: u32) -> u64 {
        self.tables[table].elements[index as usize]
    }
    #[inline(always)]
    fn set_table_ref(&mut self, table: usize, index: u32, r: u64) {
        self.tables[table].elements[index as usize] = r;
    }
    #[inline(always)]
    fn table_grow(&mut self, table: usize, delta: u32, init: u64) -> u32 {
        self.grow_table_index::<StdAlloc>(table, delta, init)
    }
    #[inline(always)]
    fn elem_len(&self, elem: usize) -> u32 {
        self.elems[elem].refs.len() as u32
    }
    #[inline(always)]
    fn elem_ref(&self, elem: usize, index: u32) -> u64 {
        self.elems[elem].refs[index as usize]
    }
    #[inline(always)]
    fn elem_drop(&mut self, elem: usize) {
        self.elems[elem].refs = Vec::new();
    }
}
impl<I, R, E> ExecutionOf<I, Val<R, E>> {
    /// Invokes a function of an abstract store synchronously, replacing
    /// `results` on success.
    pub fn invoke_in<S, P>(
        &mut self,
        store: &mut S,
        platform: &mut P,
        func: R,
        args: &[Val<R, E>],
        results: &mut Vec<Val<R, E>>,
    ) -> Result<(), ExecError>
    where
        S: AbstractStore<Instance = I, Func = R, Extern = E>,
        P: StorePlatform<S>,
    {
        interp::invoke_with::<S, P, NativeFloat, StdAlloc>(
            self, store, platform, func, args, results,
        )
    }
}
impl Execution {
    /// Invokes a function synchronously, replacing `results` on success.
    ///
    /// Host result slots are initialized to the zero value of their declared
    /// type. Hooks may reenter Wasm through a separate execution. After argument
    /// validation, invocation clears earlier frames and retains capacity for reuse.
    pub fn invoke<M: MemoryBuffer, P: Platform<M>>(
        &mut self,
        store: &mut Store<M>,
        platform: &mut P,
        func: FuncAddr,
        args: &[Value],
        results: &mut Vec<Value>,
    ) -> Result<(), ExecError> {
        interp::invoke_with::<Store<M>, P, NativeFloat, StdAlloc>(
            self, store, platform, func, args, results,
        )
    }
}

/// Validates and rewrites a complete module under the selected profile.
///
/// The returned code owns all initialization bytes and has no dependency on the
/// input slice. The default `CompileLimits` apply. Resource failures are reported
/// as `Error::Visitor(VisitError::OutOfMemory)`; use `compile_with_limits` to
/// distinguish the exhausted compiler budget from validation failures.
pub fn compile(
    bytes: &[u8],
    profile: crate::ValidationProfile,
) -> Result<ExecutableModule, crate::Error> {
    compiler::compile_default_with::<StdAlloc>(bytes, profile)
}
/// Validates and rewrites a complete module with explicit compiler budgets.
///
/// Bytecode, source maps, locals, labels, and branch-table storage are bounded;
/// compiler allocations use fallible reservation. Metadata decoding retains the
/// validator's documented allocation premise. A resource failure never yields a
/// partial executable module.
pub fn compile_with_limits(
    bytes: &[u8],
    profile: crate::ValidationProfile,
    limits: CompileLimits,
) -> Result<ExecutableModule, CompileError> {
    compiler::compile_with::<StdAlloc>(bytes, profile, limits)
}
