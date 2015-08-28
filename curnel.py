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

    _numpy_ctypes_map_inv = dict((v, k) for k, v in _numpy_ctypes_map.iteritems())

    def __init__(self, name, value):
        self.name = name
        self.value = value
        if isinstance(value, np.ndarray) and value.dtype in CurnelVariable._numpy_ctypes_map:
            self.ctype = CurnelVariable._numpy_ctypes_map[value.dtype]
            self.stype = self.ctype
            if len(value.shape) > 0:
                self.ctype += "*";
            self.shape = value.shape
        elif np.dtype(type(value)) in CurnelVariable._numpy_ctypes_map:
            self.ctype = CurnelVariable._numpy_ctypes_map[np.dtype(type(value))]
            self.stype = self.ctype
            self.shape = ()
        else:
            raise Exception("Data type %s is unsupported!" % type(value))
        

class CurnelStripModuleAndFuncDef(ast.NodeVisitor):

    def __init__(self):
        ast.NodeVisitor.__init__(self)
        self.children = []
    

    def visit_FunctionDef(self, node):
        for n in ast.iter_child_nodes(node):
            self.children.append(n)
        self.generic_visit(node)


class CurnelFindType(ast.NodeVisitor):

    def __init__(self, args, variables):
        ast.NodeVisitor.__init__(self)
        self._variables = variables
        self._args = args
        self.type = None
    

    def _get_var(self, name):
        for var in self._variables:
            if var.name == name:
                return var

        for var in self._args:
            if var.name == name:
                return var

        return None


    def visit_Name(self, node):
        var = self._get_var(node.id)
        if var != None:
            t = CurnelVariable._numpy_ctypes_map_inv[var.stype]
            self.set_type_on_visit(t)
        self.generic_visit(node)


    def visit_Attribute(self, node):
        if type(node.value) == ast.Name and node.value.id in ["threadIdx", "blockDim"] and node.attr in ["x", "y", "z"]:
            self.set_type_on_visit(np.dtype('int32'))


    def visit_Num(self, node):
        t = np.dtype(type(node.n))
        self.set_type_on_visit(t)
        self.generic_visit(node)


    def set_type_on_visit(self, t):
        if self.type == None:
            self.type = t
        elif t.itemsize > self.type.itemsize and t.kind == 'f' and self.type.kind == 'f':
            self.type = t
        elif t.itemsize > self.type.itemsize and t.kind in ('u', 'i') and self.type.kind in ('u', 'i'):
            self.type = t
        elif t.kind == 'f' and self.type.kind in ('u', 'i'):
            self.type = t


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

    def __init__(self, kfunc, args, constants):
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

        self._arg_vals = args
        self._args = []
        self._variables = []
        self._constants = [CurnelVariable(name, val) for name, val in constants.items()]

        self._ctxt_stack = deque()
        self._loop_counter = 0
        self._loop_else_stack = deque()
        self._subscript_cnt_stack = deque([0])
        self._subscript_var = None
        self.code = ""
        self.tree = ast.parse(func_string)
        self._kfunc = kfunc
        #print ast.dump(self.tree)


    def _is_name(self, name):
        return name in [var.name for var in self._variables] or \
               name in [var.name for var in self._args] or \
               name in [var.name for var in self._constants]


    def _get_var(self, name):
        for var in self._variables:
            if var.name == name:
                return var

        for var in self._args:
            if var.name == name:
                return var

        return None


    def _get_node_type(self, node):
        c = CurnelFindType(self._args, self._variables)
        c.visit(node)
        if c.type == None:
            raise Exception("Ambiguous typing; unable to resolve type.")
        return np.asarray(0, dtype=c.type)
        

    def _process_ast_node(self, cnode):
        ntype = type(cnode)

        self._ctxt_stack.append(ntype)
        
        try:
            self._node_handlers[ntype]
        except:
            raise Exception("Feature %s is unsupported by the compiler." % ntype.__name__)

        val = self._node_handlers[ntype](cnode)
        
        self._ctxt_stack.pop()

        return val
        

    def _node_string(self, node):
        global _indent
        if type(node) in CurnelPythonCCompiler._ctrl_flow_map:
            p_string = self._process_ast_node(node).replace("\n", "\n" + _indent)
            return _indent + p_string + "\n"
        else:
            return _indent + self._process_ast_node(node) + ";\n"


    def _handle_assign(self, anode):
        t_string = ""
        v_string = self._process_ast_node(anode.value)
        
        first_tar = anode.targets[0]
        if type(first_tar) == ast.Name and type(first_tar.ctx) == ast.Store:
            self._variables.append(CurnelVariable(first_tar.id, self._get_node_type(anode.value)))
            t_string += self._variables[-1].ctype + " "

        for tar in anode.targets:
            t_string += self._process_ast_node(tar) + " = "

        return t_string + v_string


    def _handle_augassign(self, aunode):
        otype = type(aunode.op)
        v_string = self._process_ast_node(aunode.value)
        t_string = self._process_ast_node(aunode.target)
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
            var = self._get_var(val_string)
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
        elif self._ctxt_stack[-2] == ast.Attribute or self._is_name(nnode.id):
            return str(nnode.id)
        else:
            raise Exception("Name '%s' not defined as an argument, locally, or as a constant for the kernel." % nnode.id)


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
        if n_string in ["threadIdx", "blockDim"] and cnode.attr in ["x", "y", "z"]:
            return n_string + "." + cnode.attr
        elif n_string == "blockIdx" and cnode.attr in ["x", "y"]:
            return n_string + "." + cnode.attr
        else:
            raise Exception("Unknown attribute or unknown variable: %s" % (n_string + "." + cnode.attr ))

    
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

        
    def compile(self):
        self.code = ""

        # Strip Module and FunctionDef nodes
        stripper = CurnelStripModuleAndFuncDef()
        stripper.visit(self.tree)

        # Extract the arguments node and the lines of CUDA to parse
        args_node = stripper.children[0]
        cu_nodes = stripper.children[1:]

        i = 0
        for arg in args_node.args:
            self._args.append(CurnelVariable(arg.id, self._arg_vals[i]))
            i += 1

        self.code += "#include <stdint.h>\n\n"

        for var in self._constants:
            self.code += "#define " + var.name + " " + str(var.value) + "\n"

        self.code += "\n__global__ void " + self._kfunc.__name__ + "("

        for var in self._args:
            self.code += var.ctype + " " + var.name + ", "
            
        self.code += "\b\b) {\n"

        for cnode in cu_nodes:
            self.code += self._node_string(cnode)

        self.code += "}\n"
        
        return self.code


# cuda decorator to specify a function as a kernel
def cuda(**kwargs):
    def cuda_decorator(kfunc):
        def kfunc_wrapper(*args):
            return CurnelPythonCCompiler(kfunc, args, kwargs).compile()
        return kfunc_wrapper
    return cuda_decorator

### Example of final code appearance
@cuda(height=2700, width=3600, range_scale=99./32.)
def mapping(cm, ss, rgbimg):
    x = threadIdx.x + blockIdx.x * blockDim.x
    y = threadIdx.y + blockIdx.y * blockDim.y
    row = height - x - 1			# flip vertically
    if x < height and y < width:
        val = ss[x,y]
        if val > 0.:				# water above 0. degrees
            maploc = val * range_scale
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

@cuda(height=2700, width=3600, range_scale=99./32.)
def test(rgbimg):
    rgbimg[y, x, z] = 200

cm_l = np.zeros((110, 3), dtype=np.uint8)
ss_l = np.zeros((2700, 3600), dtype=np.float32)
rgbimg_l = np.zeros((2700, 3600, 3), dtype=np.uint8)

print mapping(cm_l, ss_l, rgbimg_l)
