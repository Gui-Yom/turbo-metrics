use crate::{sys, CuStream};
use std::ffi::CString;
use std::path::Path;
use std::ptr::{null_mut, NonNull};
use sys::{
    cuGraphDebugDotPrint, cuGraphDestroy, cuGraphExecDestroy, cuGraphInstantiateWithFlags,
    cuGraphLaunch, CuResult,
};

pub struct CuGraph(pub(crate) NonNull<sys::CUgraph_st>);

impl CuGraph {
    pub fn dot(&self, path: impl AsRef<Path>) -> CuResult<()> {
        let path = CString::new(path.as_ref().as_os_str().as_encoded_bytes()).unwrap();
        unsafe { cuGraphDebugDotPrint(self.0.as_ptr(), path.as_ptr(), 0).result() }
    }

    pub fn instantiate(&self) -> CuResult<CuGraphExec> {
        let mut exec = null_mut();
        unsafe {
            cuGraphInstantiateWithFlags(&mut exec, self.0.as_ptr(), 0).result()?;
        }
        Ok(CuGraphExec(NonNull::new(exec).expect("null pointer")))
    }

    // pub fn get_root_nodes(&self, max_nodes: usize) -> CuResult<Vec<CuGraphNode>> {
    //     let mut nodes = Vec::with_capacity(max_nodes);
    //     nodes.extend(repeat(null_mut()).take(max_nodes));
    //     let mut len = max_nodes;
    //     unsafe {
    //         sys::cuGraphGetRootNodes(self.0.as_ptr(), nodes.as_mut_ptr(), &mut len).result()?;
    //     }
    //     dbg!(len);
    //     nodes.truncate(len);
    //     Ok(unsafe { mem::transmute(nodes) })
    // }
}

impl Drop for CuGraph {
    fn drop(&mut self) {
        unsafe { cuGraphDestroy(self.0.as_ptr()).result().unwrap() }
    }
}

pub struct CuGraphExec(NonNull<sys::CUgraphExec_st>);

impl CuGraphExec {
    pub fn launch(&self, stream: &CuStream) -> CuResult<()> {
        unsafe { cuGraphLaunch(self.0.as_ptr(), stream.0).result() }
    }
}

impl Drop for CuGraphExec {
    fn drop(&mut self) {
        unsafe { cuGraphExecDestroy(self.0.as_ptr()).result().unwrap() }
    }
}

pub struct CuGraphNode(NonNull<sys::CUgraphNode_st>);

impl CuGraphNode {
    pub fn get_type(&self) -> CuResult<sys::CUgraphNodeType> {
        let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        unsafe {
            sys::cuGraphNodeGetType(self.0.as_ptr(), &mut ty).result()?;
        }
        Ok(ty)
    }

    pub fn get_kernel_node_params(&self) -> CuResult<sys::CUDA_KERNEL_NODE_PARAMS> {
        let mut params = sys::CUDA_KERNEL_NODE_PARAMS::default();
        unsafe {
            sys::cuGraphKernelNodeGetParams_v2(self.0.as_ptr(), &mut params).result()?;
        }
        Ok(params)
    }
}
