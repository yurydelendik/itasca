//! The object model the interpreter executes against.
//!
//! The interpreter never inspects store representation directly. It resolves
//! module-local indices of an instance to handles and operates on objects
//! through those handles. Handles are opaque `Copy` values chosen by the
//! implementation, such as indices or pointers to embedder-owned objects.
//!
//! References are kept untagged in interpreter slots, tables, and element
//! segments: zero is null and any other value is produced by `func_to_ref` or
//! `extern_to_ref`.
use super::{
    ExecutableModule, HostError, InterruptAction, MemoryBuffer, Platform, Store, Val, Value,
};
use crate::FuncType;

/// The code of a function: a compiled function of an instance or a host function.
#[derive(Clone, Copy)]
pub enum Callee<I, H> {
    Wasm { instance: I, function: usize },
    Host(H),
}

/// The objects and object operations used to execute validated code.
///
/// Instance-resolution methods are only called with indices that the
/// instance's module declares, and object accessors are only called with
/// in-bounds element indices: the interpreter performs Wasm bounds checks
/// itself. Implementations must return without panicking for such inputs.
pub trait AbstractStore {
    type Instance: Copy;
    type Func: Copy;
    type Table: Copy + PartialEq;
    type Memory: Copy;
    type Global: Copy;
    type Elem: Copy;
    type Data: Copy;
    /// A non-null external reference.
    type Extern: Copy;
    /// The identity of a host function, passed to `StorePlatform::call_host`.
    type Host: Copy;
    /// A canonical signature identity: equal exactly when signatures match.
    type TypeId: Copy + PartialEq;

    /// Identifies this store for checks on reused execution state.
    fn identity(&self) -> usize;
    fn max_frames(&self) -> usize;
    fn max_values(&self) -> usize;

    /// Returns the compiled module of an instance.
    fn module(&self, instance: Self::Instance) -> &ExecutableModule;
    fn instance_func(&self, instance: Self::Instance, index: u32) -> Self::Func;
    fn instance_table(&self, instance: Self::Instance, index: u32) -> Self::Table;
    fn instance_has_memory(&self, instance: Self::Instance) -> bool;
    fn instance_memory(&self, instance: Self::Instance, index: u32) -> Self::Memory;
    fn instance_global(&self, instance: Self::Instance, index: u32) -> Self::Global;
    fn instance_elem(&self, instance: Self::Instance, index: u32) -> Self::Elem;
    fn instance_data(&self, instance: Self::Instance, index: u32) -> Self::Data;
    fn instance_type_id(&self, instance: Self::Instance, index: u32) -> Self::TypeId;

    fn callee(&self, func: Self::Func) -> Callee<Self::Instance, Self::Host>;
    fn signature(&self, func: Self::Func) -> &FuncType;
    fn func_type_id(&self, func: Self::Func) -> Self::TypeId;
    /// Whether a function handle supplied at the API boundary belongs to this store.
    fn owns_func(&self, func: Self::Func) -> bool;
    fn owns_extern(&self, r: Self::Extern) -> bool;
    /// Encodes a function as a non-zero reference.
    fn func_to_ref(&self, func: Self::Func) -> u64;
    /// Decodes a non-zero reference produced by `func_to_ref`.
    fn func_from_ref(&self, r: u64) -> Self::Func;
    fn extern_to_ref(&self, r: Self::Extern) -> u64;
    fn extern_from_ref(&self, r: u64) -> Self::Extern;

    fn memory_bytes(&self, memory: Self::Memory) -> &[u8];
    fn memory_bytes_mut(&mut self, memory: Self::Memory) -> &mut [u8];
    /// Grows by `delta` pages, returning the old page count or `u32::MAX`.
    fn memory_grow(&mut self, memory: Self::Memory, delta: u32) -> u32;
    /// Borrows a memory mutably together with a data segment's bytes.
    fn memory_and_data(&mut self, memory: Self::Memory, data: Self::Data) -> (&mut [u8], &[u8]);
    fn data_drop(&mut self, data: Self::Data);

    /// Reads the low 64 bits of a global's slot encoding.
    fn global_get(&self, global: Self::Global) -> u64;
    fn global_get_v128(&self, global: Self::Global) -> u128;
    fn global_set(&mut self, global: Self::Global, value: u64);
    fn global_set_v128(&mut self, global: Self::Global, value: u128);

    fn table_len(&self, table: Self::Table) -> u32;
    fn table_ref(&self, table: Self::Table, index: u32) -> u64;
    fn set_table_ref(&mut self, table: Self::Table, index: u32, r: u64);
    /// Grows by `delta` elements set to `init`, returning the old length or `u32::MAX`.
    fn table_grow(&mut self, table: Self::Table, delta: u32, init: u64) -> u32;

    fn elem_len(&self, elem: Self::Elem) -> u32;
    fn elem_ref(&self, elem: Self::Elem, index: u32) -> u64;
    fn elem_drop(&mut self, elem: Self::Elem);
}

/// A value at the boundary of an abstract store.
pub type StoreValue<S> = Val<<S as AbstractStore>::Func, <S as AbstractStore>::Extern>;

/// Synchronous embedding hooks for execution against an abstract store.
///
/// Every `Platform<M>` is a `StorePlatform<Store<M>>`.
pub trait StorePlatform<S: AbstractStore> {
    fn interrupt(&mut self) -> InterruptAction;
    fn call_host(
        &mut self,
        store: &mut S,
        host: S::Host,
        args: &[StoreValue<S>],
        results: &mut [StoreValue<S>],
    ) -> Result<(), HostError>;
}

impl<M: MemoryBuffer, P: Platform<M>> StorePlatform<Store<M>> for P {
    #[inline(always)]
    fn interrupt(&mut self) -> InterruptAction {
        Platform::interrupt(self)
    }
    #[inline(always)]
    fn call_host(
        &mut self,
        store: &mut Store<M>,
        host: u64,
        args: &[Value],
        results: &mut [Value],
    ) -> Result<(), HostError> {
        Platform::call_host(self, store, host, args, results)
    }
}
