import numpy as np
import ast
import inspect
import sys
from collections import deque

_indent = "    "

class CurnelVariable:
   
    # All supported numpy types.
    # Complex and float16 types not supported.
    # Default Python types also not supported.
    _numpy_ctypes_map = { \
        int                     : "long",     \
        float                   : "double",   \
        np.dtype(np.bool_)      : "uint8_t",  \
        np.dtype(np.int_)       : "long",     \
        np.dtype(np.intc)       : "int",      \
        np.dtype(np.intp)       : "ssize_t",  \
        np.dtype(np.int8)       : "int8_t",   \
        np.dtype(np.int16)      : "int16_t",  \
        np.dtype(np.int32)      : "int32_t",  \
        np.dtype(np.int64)      : "int64_t",  \
        np.dtype(np.uint8)      : "uint8_t",  \
        np.dtype(np.uint16)     : "uint16_t", \
        np.dtype(np.uint32)     : "uint32_t", \
        np.dtype(np.uint64)     : "uint64_t", \
        np.dtype(np.float_)     : "double",   \
        np.dtype(np.float32)    : "float",    \
        np.dtype(np.float64)    : "double"    \
    }

    def __init__(self, name, value):
        self.name = name
        self.value = value
        if isinstance(value, np.ndarray) and value.dtype in CurnelVariable._numpy_ctypes_map:
            self.ctype = CurnelVariable._numpy_ctypes_map[value.dtype]
            if len(value.shape) > 0:
                self.ctype += "*";
            self.shape = value.shape
        elif np.dtype(type(value)) in CurnelVariable._numpy_ctypes_map:
            self.ctype = CurnelVariable._numpy_ctypes_map[np.dtype(type(value))]
            self.shape = ()
        else:
            raise Exception("Data type %s is unsupported!" % type(value)) 


class Curnel:

    def __init__(self, kfunc):
        self._program = ""
        self._dims = None
        self._input = {}
        self._output = {}
        self._local = {}
        self._constants = {}
        self._kfunc = kfunc
        self._local["x"] = CurnelVariable("x", np.int32(0))
        self._local["y"] = CurnelVariable("y", np.int32(0))
        self._local["z"] = CurnelVariable("z", np.int32(0))

    def __get_attr__(self, name):
        if name in self._input:
            return self._input[name]
        elif name in self._output:
            return self._output[name]
        elif name in self._local:
            return self._local[name]
        elif name in self._constants:
            return self._constants[name]
        else:
            raise AttributeError()

    def getvar(self, name):
        if name in self._input:
            return self._input[name]
        elif name in self._output:
            return self._output[name]
        elif name in self._local:
            return self._local[name]
        elif name in self._constants:
            return self._constants[name]
        else:
            return None

    def generate(self):
        global _indent

        self._program += "#include <stdint.h>\n\n"

        for name, var in self._constants.items():
            self._program += "#define " + name + " " + str(var.value) + "\n"

        self._program += "\n__global__ void " + self._kfunc.__name__ + "("
        
        for name, var in self._input.items():
            self._program += "const " + var.ctype + " " + name + ", "

        for name, var in self._output.items():
            self._program += var.ctype + " " + name + ", "
            
        self._program += "\b\b) {\n"

        for name, var in self._local.items():
            self._program += _indent + var.ctype + " " + name + " = " + str(var.value) + ";\n"

        if len(self._dims) >= 1:
            self._program += _indent + "x = threadIdx.x + blockIdx.x * blockDim.x;\n"
        if len(self._dims) >= 2:
            self._program += _indent + "y = threadIdx.y + blockIdx.y * blockDim.y;\n"
        if len(self._dims) >= 3:
            self._program += _indent + "z = threadIdx.z + blockIdx.z * blockDim.z;\n"

        self._program += self._kfunc(self)

        self._program += "}\n"

        print self._program

    def put(self):
        pass
    
    def run(self):
        pass

    def get(self):
        pass

    def execute(self):
        self.generate()
        self.put()
        self.run()
        self.get()

    def setDimensions(self, *dims):
        if len(dims) < 1 or len(dims) > 3:
            raise Exception("Only 1D, 2D, and 3D data is supported by CUDA.")
        self._dims = dims

    def setInput(self, **inputvars):
        for name, value in inputvars.items():
            self._input[name] = CurnelVariable(name, value)

    def setOutput(self, **outputvars):
        for name, value in outputvars.items():
            self._output[name] = CurnelVariable(name, value)

    def setLocal(self, **localvars):
        for name, value in localvars.items():
            self._local[name] = CurnelVariable(name, value)

    def setConstants(self, **constants):
        for name, value in constants.items():
            self._constants[name] = CurnelVariable(name, value)



class CurnelStripModuleAndFuncDef(ast.NodeVisitor):

    def __init__(self):
        ast.NodeVisitor.__init__(self)
        self.children = []
    

    def visit_FunctionDef(self, node):
        for n in ast.iter_child_nodes(node):
            self.children.append(n)
        self.generic_visit(node)



class CurnelPythonCCompiler(object):

    _unop_map = {
        ast.UAdd   : '+',
        ast.USub   : '-',
        ast.Not    : '!',
        ast.Invert : '~'
    }

    _binop_map = {
        ast.Add      : '+',
        ast.Sub      : '-',
        ast.Mult     : '*',
        ast.Div      : '/',
        ast.Mod      : '%',
        ast.LShift   : '<<',
        ast.RShift   : '>>',
        ast.BitOr    : '|',
        ast.BitXor   : '^',
        ast.BitAnd   : '&'
    }

    _cmpr_map = {
        ast.Eq    : '==',
        ast.NotEq : '!=',
        ast.Lt    : '<',
        ast.LtE   : '<=',
        ast.Gt    : '>',
        ast.GtE   : '>=',
        ast.Is    : '==',
        ast.IsNot : '!=',
    }

    _ctrl_flow_map = {
        ast.If    : 'if',
        ast.For   : 'for',
        ast.While : 'while'
    }

    _const_name_map = {
        'True'  : '1',
        'False' : '0',
        'None'  : 'NULL'
    }

    def __init__(self, kernel, kfunc):
        self._node_handlers = {
            ast.Assign    : self._handle_assign,
            ast.AugAssign : self._handle_augassign,
            ast.If        : self._handle_if,
            ast.For       : self._handle_for,
            ast.While     : self._handle_while,
            ast.Break     : self._handle_break,
            ast.Continue  : self._handle_continue,
            ast.Expr      : self._handle_expr,
            ast.UnaryOp   : self._handle_unaryop,
            ast.BinOp     : self._handle_binop,
            ast.BoolOp    : self._handle_boolop,
            ast.Compare   : self._handle_cmpr,
            ast.IfExp     : self._handle_ifexp,
            ast.Attribute : self._handle_attr,
            ast.Subscript : self._handle_subscript,
            ast.Index     : self._handle_index,
            ast.Name      : self._handle_name,
            ast.Num       : self._handle_num,
            ast.Tuple     : self._handle_tuple
        }
    
        if sys.version_info >= (3, 0):
            self._node_handlers[ast.Starred] = self._handle_starred
            self._node_handlers[ast.Bytes] = self._handle_bytes
    
        if sys.version_info >= (3, 4):
            self._node_handlers[ast.NameConstant] = self._handle_name_const
    
        func_string = inspect.getsourcelines(kfunc)
        func_string = '\n'.join(func_string[0][1:])

        self._ctxt_stack = deque()
        self._loop_counter = 0
        self._loop_else_stack = deque()
        self._kernelname = ""
        self._subscript_cnt_stack = deque([0])
        self._subscript_var = None
        self.code = ""
        self.tree = ast.parse(func_string)
        #print ast.dump(self.tree)
        self.kernel = kernel


    def _node_string(self, node):
        global _indent
        if type(node) in CurnelPythonCCompiler._ctrl_flow_map:
            p_string = self._process_ast_node(node).replace("\n", "\n" + _indent)
            return _indent + p_string + "\n"
        else:
            return _indent + self._process_ast_node(node) + ";\n"


    def _handle_assign(self, anode):
        t_string = ""

        for tar in anode.targets:
            t_string += self._process_ast_node(tar) + " = "
        t_string += self._process_ast_node(anode.value)

        return t_string


    def _handle_augassign(self, aunode):
        otype = type(aunode.op)
        t_string = self._process_ast_node(aunode.target)
        v_string = self._process_ast_node(aunode.value)
        if otype == ast.FloorDiv:
            return t_string + " = floor(" + t_string + " / " + v_string + ")"
        elif otype == ast.Pow:
            return t_string + " = pow(" + t_string + ", " + v_string + ")"
        else:
            o_string = CurnelPythonCCompiler._binop_map[otype]
            return t_string + ' ' + o_string + '= ' + v_string


    def _handle_if(self, ifnode):
        t_string = self._process_ast_node(ifnode.test)
        b_string = ""
        for bnode in ifnode.body:
            b_string += self._node_string(bnode)

        elif_strings = []
        else_string = ""
        for enode in ifnode.orelse:
            if type(enode) == ast.If:
                elif_strings.append(self._process_ast_node(enode))
            else:
                else_string += self._node_string(enode)

        out_str = ""
        out_str += "if (" + t_string + ") {\n" + b_string + "}"
        for elf in elif_strings:
            out_str += " else " + elf

        if else_string != "":
            out_str += " else {\n" + else_string + "}\n" 

        return out_str


    def _handle_for(self, fornode):
        ## TODO: Figure out how to handle for loops
        return ""


    def _handle_while(self, winode):
        if winode.orelse:
            has_else = True
        else:
            has_else = False

        self._loop_counter += 1
        self._loop_else_stack.append((has_else, self._loop_counter))

        t_string = self._process_ast_node(winode.test)
        b_string = ""
        for bnode in winode.body:
            b_string += self._node_string(bnode)

        else_string = ""
        for enode in winode.orelse:
            else_string += self._node_string(enode)

        out_str = ""

        if has_else:
            out_str += "int did_break_" + str(self._loop_counter) + " = 0;\n"

        out_str += "while (" + t_string + ") {\n" + b_string + "}\n"

        if has_else:
            out_str += "if (!did_break_" + str(self._loop_counter) + ") {\n" + else_string + "}\n"
        
        self._loop_else_stack.pop()

        return out_str


    def _handle_binop(self, binode):
        l_string = self._process_ast_node(binode.left)
        r_string = self._process_ast_node(binode.right)
        otype = type(binode.op)
        if otype == ast.FloorDiv:
            return "floor(" + l_string + " / " + r_string + ")"
        elif otype == ast.Pow:
            return "pow(" + l_string + ", " + r_string + ")"
        else:
            o_string = CurnelPythonCCompiler._binop_map[otype]
            return "(" + l_string + o_string + r_string + ")"


    def _handle_boolop(self, boolop):
        otype = type(boolop.op)
        out_str = ""
        
        if otype == ast.And:
            o_string = " && "
        else:
            o_string = " || "

        out_str += self._process_ast_node(boolop.values[0])
        for val in boolop.values[1:]:
            out_str += o_string + self._process_ast_node(val)
            
        return '(' + out_str + ')'
            

    def _handle_cmpr(self, cmprop):
        op_strings = []
        for op in cmprop.ops:
            if type(op) == ast.In or type(op) == ast.NotIn:
                raise Exception("in and not in keywords currently are not supported.")
            op_strings.append(CurnelPythonCCompiler._cmpr_map[type(op)])

        val_strings = []

        val_strings.append(self._process_ast_node(cmprop.left))
        for val in cmprop.comparators:
            val_strings.append(self._process_ast_node(val)) 

        for i in range(len(val_strings)-1):
            ## Compare pointers on Is and IsNot
            if op_strings[i] == ast.Is or op_strings[i] == ast.IsNot:
                val_strings[i] = '(&' + val_strings[i] + ' ' + op_strings[i] + ' &' + val_strings[i+1] + ')'
            else:
                val_strings[i] = '(' + val_strings[i] + ' ' + op_strings[i] + ' ' + val_strings[i+1] + ')'

        return ' && '.join(val_strings[:-1])


    def _handle_ifexp(self, ienode):
        sif_string = "(" + self._process_ast_node(ienode.test) + ") ? "
        sif_string += "(" + self._process_ast_node(ienode.body) + ") : "
        sif_string += "(" + self._process_ast_node(ienode.orelse) + ")"
        return sif_string

    
    def _handle_subscript(self, ssnode):
        sli_list = self._process_ast_node(ssnode.slice)

        if not isinstance(sli_list, list):
            sli_list = [sli_list]
        subscripts = len(sli_list)

        self._subscript_cnt_stack[-1] += subscripts

        val_string = self._process_ast_node(ssnode.value)

        out_string = val_string
        sli_string = ""

        if type(ssnode.value) != ast.Subscript:
            var = self.kernel.getvar(val_string)
            if var == None:
                raise Exception("Name '%s' not defined as input, output, locally, or as a constant for the kernel." % val_string)
            elif len(var.shape) < self._subscript_cnt_stack[-1]:
                print self._subscript_cnt_stack
                raise Exception("Name '%s' cannot be subscripted %d time(s)." % (var.name, self._subscript_cnt_stack[-1]))
            self._subscript_var = var 
            out_string += "["

        elif self._subscript_cnt_stack[-1] > 3:
            raise Exception("Subscripting more than 3 times is not supported.")

        dims = len(self._subscript_var.shape)
        for i in range(subscripts):
            self._subscript_cnt_stack[-1] -= 1
            for j in range(self._subscript_cnt_stack[-1]):
                sli_list[i] += " * " + str(self._subscript_var.shape[dims-j-1])
        sli_string = ' + '.join(sli_list)

        if self._subscript_cnt_stack[-1] > 0:
            out_string += sli_string
        else: 
            self._subscript_var = None
            out_string += sli_string + "]"


        return out_string


    def _handle_name(self, nnode):
        if nnode.id in CurnelPythonCCompiler._const_name_map:
            return CurnelPythonCCompiler._const_name_map[nnode.id]
        if self.kernel.getvar(nnode.id) == None:
            raise Exception("Name '%s' not defined as input, output, locally, or as a constant for the kernel." % nnode.id)
        else:
            return str(nnode.id)


    def _handle_name_const(self, ncnode):
        return ""


    # Only handles packing/unpacking and assignment contexts!
    def _handle_tuple(self, tnode):
        if self._ctxt_stack[-2] in (ast.Assign, ast.For, ast.Index):
            strings = []
            for elem in tnode.elts:
                strings.append(self._process_ast_node(elem))
            return strings
        else:
            raise Exception("Tuple support is only available in non-nested packing/unpacking and assignment contexts.")

            
    def _handle_break(self, cnode):
        if self._loop_else_stack[-1][0]:
            return "did_break_" + str(self._loop_else_stack[-1][1]) + " = 1; break"
        else:
            return "break"

            
    def _handle_continue(self, cnode):
        return "continue"
        
        
    def _handle_expr(self, cnode):
        return self._process_ast_node(cnode.value)
        
        
    def _handle_unaryop(self, cnode):
        return CurnelPythonCCompiler._unop_map[type(cnode.op)] + '(' + self._process_ast_node(cnode.operand) + ')'
        
    
    def _handle_attr(self, cnode):
        n_string = self._process_ast_node(cnode.value)
        if n_string != self._kernelname:
            raise Exception("Dot operator not supported.")
        return str(cnode.attr)
    
    
    def _handle_index(self, cnode):
        self._subscript_cnt_stack.append(0)
        out_string = self._process_ast_node(cnode.value)
        self._subscript_cnt_stack.pop()
        return out_string
       
       
    def _handle_starred(self, cnode):
        n_string = self._process_ast_node(cnode.value)
        return '&(' + n_string + ')'
    
    
    def _handle_num(self, cnode):
        return str(cnode.n)
    
    
    def _handle_str(self, cnode):
        return '"' + cnode.s + '"'
    
    
    def _handle_bytes(self, cnode):
        return '"' + cnode.s + '"'
    
    
    def _process_ast_node(self, cnode):
        ntype = type(cnode)

        self._ctxt_stack.append(ntype)
        
        try:
            return self._node_handlers[ntype](cnode)
        except:
            raise Exception("Feature %s is unsupported by the compiler." % ntype.__name__)
        
        self._ctxt_stack.pop()

        
    def compile(self):
        self.code = ""

        # Strip Module and FunctionDef nodes
        stripper = CurnelStripModuleAndFuncDef()
        stripper.visit(self.tree)

        # Extract the arguments node and the lines of CUDA to parse
        args_node = stripper.children[0]
        cu_nodes = stripper.children[1:]

        # Make sure we only have 1 argument to the function
        assert len(args_node.args) == 1
        args_node.args[0].arg = self._kernelname

        for cnode in cu_nodes:
            self.code += self._node_string(cnode)
            
        return self.code


# cuda decorator to specify a function as a kernel
def cuda_kernel(kfunc):
    def kfunc_wrapper(kernel):
        return CurnelPythonCCompiler(kernel, kfunc).compile()
    return kfunc_wrapper



### Example of final code appearance
@cuda_kernel
def mapping(k):
    row = HEIGHT - x - 1			# flip vertically
    if x < HEIGHT and y < WIDTH:
        val = ss[x,y]
        if val > 0.:				# water above 0. degrees
            maploc = val * RANGE_SCALE
            rgbimg[row,x,0] = cm[maploc,0]
            rgbimg[row,x,1] = cm[maploc,1]
            rgbimg[row,x,2] = cm[maploc,2]
        elif val <= -100.:			# color it black for land
            rgbimg[row,x,0] = 32
            rgbimg[row,x,1] = 32
            rgbimg[row,x,2] = 32
        elif val > -2. and val <= 0.:  # water between -2. and 0. degrees
              # blend from white to magenta
            d = -125. * val
            rgbimg[row,x,0] = 255
            rgbimg[row,x,1] = d
            rgbimg[row,x,2] = 255
        else:					# frozen water, gray
            rgbimg[row,x,0] = 190
            rgbimg[row,x,1] = 190
            rgbimg[row,x,2] = 190

@cuda_kernel
def test(k):
    rgbimg[y, x, z] = 200

cm_l = np.zeros((110, 3), dtype=np.uint8)
ss_l = np.zeros((2700, 3600), dtype=np.float32)
rgbimg_l = np.zeros((2700, 3600, 3), dtype=np.uint8)

k = Curnel(mapping)
k.setDimensions(2700, 3600)
k.setInput(cm=cm_l, ss=ss_l)
k.setConstants(WIDTH=3600, HEIGHT=2700, RANGE_SCALE=99./32.)
k.setLocal(row=0, maploc=0, d=0.0, val=0.0)
k.setOutput(rgbimg=rgbimg_l)
k.execute()

