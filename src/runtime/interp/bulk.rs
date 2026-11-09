//! Bulk memory and table operations and segment drops.
//!
//! Active segment initialization is compiled into these instructions, so it
//! shares their bounds checks.
use super::*;

/// Executes a bulk instruction. Fill, copy, and init read their destination,
/// source or value, and length from three consecutive slots starting at `d`.
pub(super) fn execute<S: AbstractStore>(
    store: &mut S,
    inst: S::Instance,
    function: usize,
    base: usize,
    slots: &[u64],
    instruction: Instr,
) -> Result<(), ExecError> {
    match instruction {
        Instr::DataDrop { data } => {
            let address = store.instance_data(inst, data);
            store.data_drop(address);
        }
        Instr::ElemDrop { elem } => {
            let address = store.instance_elem(inst, elem);
            store.elem_drop(address);
        }
        Instr::MemoryFill { d } | Instr::MemoryCopy { d } | Instr::MemoryInit { d, .. } => {
            let destination = slots[base + d as usize] as u32 as u64;
            let source = slots[base + d as usize + 1] as u32;
            let len = slots[base + d as usize + 2] as u32 as u64;
            let memory = store.instance_memory(inst, 0);
            let bytes = store.memory_bytes(memory).len();
            let dst = range(destination, len, bytes, Trap::MemoryOutOfBounds)?;
            match instruction {
                Instr::MemoryInit { data, .. } => {
                    let address = store.instance_data(inst, data);
                    let (target, data) = store.memory_and_data(memory, address);
                    let src = range(source as u64, len, data.len(), Trap::MemoryOutOfBounds)?;
                    target[dst].copy_from_slice(&data[src]);
                }
                Instr::MemoryCopy { .. } => {
                    let src = range(source as u64, len, bytes, Trap::MemoryOutOfBounds)?;
                    let target = store.memory_bytes_mut(memory);
                    if dst.end <= src.start {
                        let (left, right) = target.split_at_mut(src.start);
                        left[dst].copy_from_slice(&right[..len as usize]);
                    } else if src.end <= dst.start {
                        let (left, right) = target.split_at_mut(dst.start);
                        right[..len as usize].copy_from_slice(&left[src]);
                    } else if dst.start <= src.start {
                        let mut i = 0;
                        while i < len as usize {
                            target[dst.start + i] = target[src.start + i];
                            i += 1;
                        }
                    } else {
                        let mut i = len as usize;
                        while i > 0 {
                            i -= 1;
                            target[dst.start + i] = target[src.start + i];
                        }
                    }
                }
                _ => {
                    let target = store.memory_bytes_mut(memory);
                    target[dst].fill(source as u8);
                }
            }
        }
        Instr::TableCopy { d, entry } | Instr::TableInit { d, entry } => {
            let (a, b) = store.module(inst).functions[function].indirect[entry as usize];
            let destination = slots[base + d as usize] as u32 as u64;
            let source = slots[base + d as usize + 1] as u32 as u64;
            let len = slots[base + d as usize + 2] as u32 as u64;
            let table_index = if matches!(instruction, Instr::TableInit { .. }) {
                b
            } else {
                a
            };
            let table = store.instance_table(inst, table_index);
            let dst = range(
                destination,
                len,
                store.table_len(table) as usize,
                Trap::TableOutOfBounds,
            )?;
            if matches!(instruction, Instr::TableInit { .. }) {
                let elem = store.instance_elem(inst, a);
                let src = range(
                    source,
                    len,
                    store.elem_len(elem) as usize,
                    Trap::TableOutOfBounds,
                )?;
                let mut i = 0;
                while i < len as usize {
                    let r = store.elem_ref(elem, (src.start + i) as u32);
                    store.set_table_ref(table, (dst.start + i) as u32, r);
                    i += 1;
                }
            } else {
                let src_table = store.instance_table(inst, b);
                let src = range(
                    source,
                    len,
                    store.table_len(src_table) as usize,
                    Trap::TableOutOfBounds,
                )?;
                if table == src_table && dst.start > src.start {
                    let mut i = len as usize;
                    while i > 0 {
                        i -= 1;
                        let r = store.table_ref(src_table, (src.start + i) as u32);
                        store.set_table_ref(table, (dst.start + i) as u32, r);
                    }
                } else {
                    let mut i = 0;
                    while i < len as usize {
                        let r = store.table_ref(src_table, (src.start + i) as u32);
                        store.set_table_ref(table, (dst.start + i) as u32, r);
                        i += 1;
                    }
                }
            }
        }
        Instr::TableFill { d, table } => {
            let destination = slots[base + d as usize] as u32 as u64;
            let value = slots[base + d as usize + 1];
            let len = slots[base + d as usize + 2] as u32 as u64;
            let t = store.instance_table(inst, table);
            let dst = range(
                destination,
                len,
                store.table_len(t) as usize,
                Trap::TableOutOfBounds,
            )?;
            let mut i = dst.start;
            while i < dst.end {
                store.set_table_ref(t, i as u32, value);
                i += 1;
            }
        }
        _ => return Err(ExecError::Internal),
    }
    Ok(())
}
