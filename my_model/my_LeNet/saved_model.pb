Üï
Ñ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878µ¤

le_net5/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namele_net5/conv2d/kernel

)le_net5/conv2d/kernel/Read/ReadVariableOpReadVariableOple_net5/conv2d/kernel*&
_output_shapes
:*
dtype0
~
le_net5/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namele_net5/conv2d/bias
w
'le_net5/conv2d/bias/Read/ReadVariableOpReadVariableOple_net5/conv2d/bias*
_output_shapes
:*
dtype0

le_net5/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namele_net5/conv2d_1/kernel

+le_net5/conv2d_1/kernel/Read/ReadVariableOpReadVariableOple_net5/conv2d_1/kernel*&
_output_shapes
:*
dtype0

le_net5/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namele_net5/conv2d_1/bias
{
)le_net5/conv2d_1/bias/Read/ReadVariableOpReadVariableOple_net5/conv2d_1/bias*
_output_shapes
:*
dtype0

le_net5/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x*%
shared_namele_net5/dense/kernel
~
(le_net5/dense/kernel/Read/ReadVariableOpReadVariableOple_net5/dense/kernel*
_output_shapes
:	x*
dtype0
|
le_net5/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*#
shared_namele_net5/dense/bias
u
&le_net5/dense/bias/Read/ReadVariableOpReadVariableOple_net5/dense/bias*
_output_shapes
:x*
dtype0

le_net5/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*'
shared_namele_net5/dense_1/kernel

*le_net5/dense_1/kernel/Read/ReadVariableOpReadVariableOple_net5/dense_1/kernel*
_output_shapes

:xT*
dtype0

le_net5/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*%
shared_namele_net5/dense_1/bias
y
(le_net5/dense_1/bias/Read/ReadVariableOpReadVariableOple_net5/dense_1/bias*
_output_shapes
:T*
dtype0

le_net5/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T
*'
shared_namele_net5/dense_2/kernel

*le_net5/dense_2/kernel/Read/ReadVariableOpReadVariableOple_net5/dense_2/kernel*
_output_shapes

:T
*
dtype0

le_net5/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_namele_net5/dense_2/bias
y
(le_net5/dense_2/bias/Read/ReadVariableOpReadVariableOple_net5/dense_2/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/le_net5/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/le_net5/conv2d/kernel/m

0Adam/le_net5/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/le_net5/conv2d/kernel/m*&
_output_shapes
:*
dtype0

Adam/le_net5/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/le_net5/conv2d/bias/m

.Adam/le_net5/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/le_net5/conv2d/bias/m*
_output_shapes
:*
dtype0
 
Adam/le_net5/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/le_net5/conv2d_1/kernel/m

2Adam/le_net5/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/le_net5/conv2d_1/kernel/m*&
_output_shapes
:*
dtype0

Adam/le_net5/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/le_net5/conv2d_1/bias/m

0Adam/le_net5/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/le_net5/conv2d_1/bias/m*
_output_shapes
:*
dtype0

Adam/le_net5/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x*,
shared_nameAdam/le_net5/dense/kernel/m

/Adam/le_net5/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/le_net5/dense/kernel/m*
_output_shapes
:	x*
dtype0

Adam/le_net5/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x**
shared_nameAdam/le_net5/dense/bias/m

-Adam/le_net5/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/le_net5/dense/bias/m*
_output_shapes
:x*
dtype0

Adam/le_net5/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*.
shared_nameAdam/le_net5/dense_1/kernel/m

1Adam/le_net5/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/le_net5/dense_1/kernel/m*
_output_shapes

:xT*
dtype0

Adam/le_net5/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*,
shared_nameAdam/le_net5/dense_1/bias/m

/Adam/le_net5/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/le_net5/dense_1/bias/m*
_output_shapes
:T*
dtype0

Adam/le_net5/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T
*.
shared_nameAdam/le_net5/dense_2/kernel/m

1Adam/le_net5/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/le_net5/dense_2/kernel/m*
_output_shapes

:T
*
dtype0

Adam/le_net5/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameAdam/le_net5/dense_2/bias/m

/Adam/le_net5/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/le_net5/dense_2/bias/m*
_output_shapes
:
*
dtype0

Adam/le_net5/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/le_net5/conv2d/kernel/v

0Adam/le_net5/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/le_net5/conv2d/kernel/v*&
_output_shapes
:*
dtype0

Adam/le_net5/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/le_net5/conv2d/bias/v

.Adam/le_net5/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/le_net5/conv2d/bias/v*
_output_shapes
:*
dtype0
 
Adam/le_net5/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/le_net5/conv2d_1/kernel/v

2Adam/le_net5/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/le_net5/conv2d_1/kernel/v*&
_output_shapes
:*
dtype0

Adam/le_net5/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/le_net5/conv2d_1/bias/v

0Adam/le_net5/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/le_net5/conv2d_1/bias/v*
_output_shapes
:*
dtype0

Adam/le_net5/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x*,
shared_nameAdam/le_net5/dense/kernel/v

/Adam/le_net5/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/le_net5/dense/kernel/v*
_output_shapes
:	x*
dtype0

Adam/le_net5/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x**
shared_nameAdam/le_net5/dense/bias/v

-Adam/le_net5/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/le_net5/dense/bias/v*
_output_shapes
:x*
dtype0

Adam/le_net5/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*.
shared_nameAdam/le_net5/dense_1/kernel/v

1Adam/le_net5/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/le_net5/dense_1/kernel/v*
_output_shapes

:xT*
dtype0

Adam/le_net5/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*,
shared_nameAdam/le_net5/dense_1/bias/v

/Adam/le_net5/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/le_net5/dense_1/bias/v*
_output_shapes
:T*
dtype0

Adam/le_net5/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T
*.
shared_nameAdam/le_net5/dense_2/kernel/v

1Adam/le_net5/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/le_net5/dense_2/kernel/v*
_output_shapes

:T
*
dtype0

Adam/le_net5/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameAdam/le_net5/dense_2/bias/v

/Adam/le_net5/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/le_net5/dense_2/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
ç9
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¢9
value9B9 B9
¶
c1
p1
c2
p2
flatten
f1
f2
f3
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
 trainable_variables
!	variables
"	keras_api
R
#regularization_losses
$trainable_variables
%	variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0trainable_variables
1	variables
2	keras_api
h

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
þ
9iter

:beta_1

;beta_2
	<decay
=learning_ratemvmwmxmy'mz(m{-m|.m}3m~4mvvvv'v(v-v.v3v4v
 
F
0
1
2
3
'4
(5
-6
.7
38
49
F
0
1
2
3
'4
(5
-6
.7
38
49
­

regularization_losses
>layer_regularization_losses
?layer_metrics

@layers
Ametrics
trainable_variables
	variables
Bnon_trainable_variables
 
OM
VARIABLE_VALUEle_net5/conv2d/kernel$c1/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEle_net5/conv2d/bias"c1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
Clayer_regularization_losses
Dlayer_metrics

Elayers
Fmetrics
trainable_variables
	variables
Gnon_trainable_variables
 
 
 
­
regularization_losses
Hlayer_regularization_losses
Ilayer_metrics

Jlayers
Kmetrics
trainable_variables
	variables
Lnon_trainable_variables
QO
VARIABLE_VALUEle_net5/conv2d_1/kernel$c2/kernel/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEle_net5/conv2d_1/bias"c2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
Mlayer_regularization_losses
Nlayer_metrics

Olayers
Pmetrics
trainable_variables
	variables
Qnon_trainable_variables
 
 
 
­
regularization_losses
Rlayer_regularization_losses
Slayer_metrics

Tlayers
Umetrics
 trainable_variables
!	variables
Vnon_trainable_variables
 
 
 
­
#regularization_losses
Wlayer_regularization_losses
Xlayer_metrics

Ylayers
Zmetrics
$trainable_variables
%	variables
[non_trainable_variables
NL
VARIABLE_VALUEle_net5/dense/kernel$f1/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEle_net5/dense/bias"f1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
­
)regularization_losses
\layer_regularization_losses
]layer_metrics

^layers
_metrics
*trainable_variables
+	variables
`non_trainable_variables
PN
VARIABLE_VALUEle_net5/dense_1/kernel$f2/kernel/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEle_net5/dense_1/bias"f2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
­
/regularization_losses
alayer_regularization_losses
blayer_metrics

clayers
dmetrics
0trainable_variables
1	variables
enon_trainable_variables
PN
VARIABLE_VALUEle_net5/dense_2/kernel$f3/kernel/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEle_net5/dense_2/bias"f3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
­
5regularization_losses
flayer_regularization_losses
glayer_metrics

hlayers
imetrics
6trainable_variables
7	variables
jnon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
8
0
1
2
3
4
5
6
7

k0
l1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	mtotal
	ncount
o	variables
p	keras_api
D
	qtotal
	rcount
s
_fn_kwargs
t	variables
u	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

m0
n1

o	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

q0
r1

t	variables
rp
VARIABLE_VALUEAdam/le_net5/conv2d/kernel/m@c1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/le_net5/conv2d/bias/m>c1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/le_net5/conv2d_1/kernel/m@c2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/le_net5/conv2d_1/bias/m>c2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/le_net5/dense/kernel/m@f1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/le_net5/dense/bias/m>f1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/le_net5/dense_1/kernel/m@f2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/le_net5/dense_1/bias/m>f2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/le_net5/dense_2/kernel/m@f3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/le_net5/dense_2/bias/m>f3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/le_net5/conv2d/kernel/v@c1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/le_net5/conv2d/bias/v>c1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/le_net5/conv2d_1/kernel/v@c2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/le_net5/conv2d_1/bias/v>c2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/le_net5/dense/kernel/v@f1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/le_net5/dense/bias/v>f1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/le_net5/dense_1/kernel/v@f2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/le_net5/dense_1/bias/v>f2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/le_net5/dense_2/kernel/v@f3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/le_net5/dense_2/bias/v>f3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ  
ª
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1le_net5/conv2d/kernelle_net5/conv2d/biasle_net5/conv2d_1/kernelle_net5/conv2d_1/biasle_net5/dense/kernelle_net5/dense/biasle_net5/dense_1/kernelle_net5/dense_1/biasle_net5/dense_2/kernelle_net5/dense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_38898
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ç
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)le_net5/conv2d/kernel/Read/ReadVariableOp'le_net5/conv2d/bias/Read/ReadVariableOp+le_net5/conv2d_1/kernel/Read/ReadVariableOp)le_net5/conv2d_1/bias/Read/ReadVariableOp(le_net5/dense/kernel/Read/ReadVariableOp&le_net5/dense/bias/Read/ReadVariableOp*le_net5/dense_1/kernel/Read/ReadVariableOp(le_net5/dense_1/bias/Read/ReadVariableOp*le_net5/dense_2/kernel/Read/ReadVariableOp(le_net5/dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp0Adam/le_net5/conv2d/kernel/m/Read/ReadVariableOp.Adam/le_net5/conv2d/bias/m/Read/ReadVariableOp2Adam/le_net5/conv2d_1/kernel/m/Read/ReadVariableOp0Adam/le_net5/conv2d_1/bias/m/Read/ReadVariableOp/Adam/le_net5/dense/kernel/m/Read/ReadVariableOp-Adam/le_net5/dense/bias/m/Read/ReadVariableOp1Adam/le_net5/dense_1/kernel/m/Read/ReadVariableOp/Adam/le_net5/dense_1/bias/m/Read/ReadVariableOp1Adam/le_net5/dense_2/kernel/m/Read/ReadVariableOp/Adam/le_net5/dense_2/bias/m/Read/ReadVariableOp0Adam/le_net5/conv2d/kernel/v/Read/ReadVariableOp.Adam/le_net5/conv2d/bias/v/Read/ReadVariableOp2Adam/le_net5/conv2d_1/kernel/v/Read/ReadVariableOp0Adam/le_net5/conv2d_1/bias/v/Read/ReadVariableOp/Adam/le_net5/dense/kernel/v/Read/ReadVariableOp-Adam/le_net5/dense/bias/v/Read/ReadVariableOp1Adam/le_net5/dense_1/kernel/v/Read/ReadVariableOp/Adam/le_net5/dense_1/bias/v/Read/ReadVariableOp1Adam/le_net5/dense_2/kernel/v/Read/ReadVariableOp/Adam/le_net5/dense_2/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_39149
Ö	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamele_net5/conv2d/kernelle_net5/conv2d/biasle_net5/conv2d_1/kernelle_net5/conv2d_1/biasle_net5/dense/kernelle_net5/dense/biasle_net5/dense_1/kernelle_net5/dense_1/biasle_net5/dense_2/kernelle_net5/dense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/le_net5/conv2d/kernel/mAdam/le_net5/conv2d/bias/mAdam/le_net5/conv2d_1/kernel/mAdam/le_net5/conv2d_1/bias/mAdam/le_net5/dense/kernel/mAdam/le_net5/dense/bias/mAdam/le_net5/dense_1/kernel/mAdam/le_net5/dense_1/bias/mAdam/le_net5/dense_2/kernel/mAdam/le_net5/dense_2/bias/mAdam/le_net5/conv2d/kernel/vAdam/le_net5/conv2d/bias/vAdam/le_net5/conv2d_1/kernel/vAdam/le_net5/conv2d_1/bias/vAdam/le_net5/dense/kernel/vAdam/le_net5/dense/bias/vAdam/le_net5/dense_1/kernel/vAdam/le_net5/dense_1/bias/vAdam/le_net5/dense_2/kernel/vAdam/le_net5/dense_2/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_39276ú
º
^
B__inference_flatten_layer_call_and_return_conditional_losses_38747

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_38675

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
çQ
»
__inference__traced_save_39149
file_prefix4
0savev2_le_net5_conv2d_kernel_read_readvariableop2
.savev2_le_net5_conv2d_bias_read_readvariableop6
2savev2_le_net5_conv2d_1_kernel_read_readvariableop4
0savev2_le_net5_conv2d_1_bias_read_readvariableop3
/savev2_le_net5_dense_kernel_read_readvariableop1
-savev2_le_net5_dense_bias_read_readvariableop5
1savev2_le_net5_dense_1_kernel_read_readvariableop3
/savev2_le_net5_dense_1_bias_read_readvariableop5
1savev2_le_net5_dense_2_kernel_read_readvariableop3
/savev2_le_net5_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop;
7savev2_adam_le_net5_conv2d_kernel_m_read_readvariableop9
5savev2_adam_le_net5_conv2d_bias_m_read_readvariableop=
9savev2_adam_le_net5_conv2d_1_kernel_m_read_readvariableop;
7savev2_adam_le_net5_conv2d_1_bias_m_read_readvariableop:
6savev2_adam_le_net5_dense_kernel_m_read_readvariableop8
4savev2_adam_le_net5_dense_bias_m_read_readvariableop<
8savev2_adam_le_net5_dense_1_kernel_m_read_readvariableop:
6savev2_adam_le_net5_dense_1_bias_m_read_readvariableop<
8savev2_adam_le_net5_dense_2_kernel_m_read_readvariableop:
6savev2_adam_le_net5_dense_2_bias_m_read_readvariableop;
7savev2_adam_le_net5_conv2d_kernel_v_read_readvariableop9
5savev2_adam_le_net5_conv2d_bias_v_read_readvariableop=
9savev2_adam_le_net5_conv2d_1_kernel_v_read_readvariableop;
7savev2_adam_le_net5_conv2d_1_bias_v_read_readvariableop:
6savev2_adam_le_net5_dense_kernel_v_read_readvariableop8
4savev2_adam_le_net5_dense_bias_v_read_readvariableop<
8savev2_adam_le_net5_dense_1_kernel_v_read_readvariableop:
6savev2_adam_le_net5_dense_1_bias_v_read_readvariableop<
8savev2_adam_le_net5_dense_2_kernel_v_read_readvariableop:
6savev2_adam_le_net5_dense_2_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_7f305baa42d244529e27cd3b954e8867/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameä
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*ö
valueìBé(B$c1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c1/bias/.ATTRIBUTES/VARIABLE_VALUEB$c2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c2/bias/.ATTRIBUTES/VARIABLE_VALUEB$f1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"f1/bias/.ATTRIBUTES/VARIABLE_VALUEB$f2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"f2/bias/.ATTRIBUTES/VARIABLE_VALUEB$f3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"f3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB@c1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>c1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@c2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>c2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@f1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>f1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@f2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>f2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@f3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>f3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@c1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>c1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@c2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>c2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@f1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>f1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@f2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>f2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@f3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>f3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesØ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_le_net5_conv2d_kernel_read_readvariableop.savev2_le_net5_conv2d_bias_read_readvariableop2savev2_le_net5_conv2d_1_kernel_read_readvariableop0savev2_le_net5_conv2d_1_bias_read_readvariableop/savev2_le_net5_dense_kernel_read_readvariableop-savev2_le_net5_dense_bias_read_readvariableop1savev2_le_net5_dense_1_kernel_read_readvariableop/savev2_le_net5_dense_1_bias_read_readvariableop1savev2_le_net5_dense_2_kernel_read_readvariableop/savev2_le_net5_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop7savev2_adam_le_net5_conv2d_kernel_m_read_readvariableop5savev2_adam_le_net5_conv2d_bias_m_read_readvariableop9savev2_adam_le_net5_conv2d_1_kernel_m_read_readvariableop7savev2_adam_le_net5_conv2d_1_bias_m_read_readvariableop6savev2_adam_le_net5_dense_kernel_m_read_readvariableop4savev2_adam_le_net5_dense_bias_m_read_readvariableop8savev2_adam_le_net5_dense_1_kernel_m_read_readvariableop6savev2_adam_le_net5_dense_1_bias_m_read_readvariableop8savev2_adam_le_net5_dense_2_kernel_m_read_readvariableop6savev2_adam_le_net5_dense_2_bias_m_read_readvariableop7savev2_adam_le_net5_conv2d_kernel_v_read_readvariableop5savev2_adam_le_net5_conv2d_bias_v_read_readvariableop9savev2_adam_le_net5_conv2d_1_kernel_v_read_readvariableop7savev2_adam_le_net5_conv2d_1_bias_v_read_readvariableop6savev2_adam_le_net5_dense_kernel_v_read_readvariableop4savev2_adam_le_net5_dense_bias_v_read_readvariableop8savev2_adam_le_net5_dense_1_kernel_v_read_readvariableop6savev2_adam_le_net5_dense_1_bias_v_read_readvariableop8savev2_adam_le_net5_dense_2_kernel_v_read_readvariableop6savev2_adam_le_net5_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Î
_input_shapes¼
¹: :::::	x:x:xT:T:T
:
: : : : : : : : : :::::	x:x:xT:T:T
:
:::::	x:x:xT:T:T
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	x: 

_output_shapes
:x:$ 

_output_shapes

:xT: 

_output_shapes
:T:$	 

_output_shapes

:T
: 


_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	x: 

_output_shapes
:x:$ 

_output_shapes

:xT: 

_output_shapes
:T:$ 

_output_shapes

:T
: 

_output_shapes
:
:,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
::%"!

_output_shapes
:	x: #

_output_shapes
:x:$$ 

_output_shapes

:xT: %

_output_shapes
:T:$& 

_output_shapes

:T
: '

_output_shapes
:
:(

_output_shapes
: 
£
C
'__inference_flatten_layer_call_fn_38949

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_387472
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
ª
B__inference_dense_1_layer_call_and_return_conditional_losses_38793

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿx:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
	
©
A__inference_conv2d_layer_call_and_return_conditional_losses_38909

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidg
IdentityIdentitySigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ  :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ª
I
-__inference_max_pooling2d_layer_call_fn_38669

inputs
identityì
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_386632
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
$
Ä
B__inference_le_net5_layer_call_and_return_conditional_losses_38837
input_1
conv2d_38707
conv2d_38709
conv2d_1_38735
conv2d_1_38737
dense_38777
dense_38779
dense_1_38804
dense_1_38806
dense_2_38831
dense_2_38833
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_38707conv2d_38709*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_386962 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_386632
max_pooling2d/PartitionedCall¼
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_38735conv2d_1_38737*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_387242"
 conv2d_1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_386752!
max_pooling2d_1/PartitionedCallö
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_387472
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_38777dense_38779*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_387662
dense/StatefulPartitionedCall¯
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_38804dense_1_38806*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_387932!
dense_1/StatefulPartitionedCall±
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_38831dense_2_38833*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_388202!
dense_2/StatefulPartitionedCall¤
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ  ::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1
þ
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_38663

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
|
'__inference_dense_2_layer_call_fn_39009

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_388202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿT::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
	
«
C__inference_conv2d_1_layer_call_and_return_conditional_losses_38724

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2	
Sigmoidg
IdentityIdentitySigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
{
&__inference_conv2d_layer_call_fn_38918

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_386962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
¯
ª
B__inference_dense_2_layer_call_and_return_conditional_losses_39000

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿT:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
º
^
B__inference_flatten_layer_call_and_return_conditional_losses_38944

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
©
A__inference_conv2d_layer_call_and_return_conditional_losses_38696

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidg
IdentityIdentitySigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ  :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
³£
É
!__inference__traced_restore_39276
file_prefix*
&assignvariableop_le_net5_conv2d_kernel*
&assignvariableop_1_le_net5_conv2d_bias.
*assignvariableop_2_le_net5_conv2d_1_kernel,
(assignvariableop_3_le_net5_conv2d_1_bias+
'assignvariableop_4_le_net5_dense_kernel)
%assignvariableop_5_le_net5_dense_bias-
)assignvariableop_6_le_net5_dense_1_kernel+
'assignvariableop_7_le_net5_dense_1_bias-
)assignvariableop_8_le_net5_dense_2_kernel+
'assignvariableop_9_le_net5_dense_2_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_14
0assignvariableop_19_adam_le_net5_conv2d_kernel_m2
.assignvariableop_20_adam_le_net5_conv2d_bias_m6
2assignvariableop_21_adam_le_net5_conv2d_1_kernel_m4
0assignvariableop_22_adam_le_net5_conv2d_1_bias_m3
/assignvariableop_23_adam_le_net5_dense_kernel_m1
-assignvariableop_24_adam_le_net5_dense_bias_m5
1assignvariableop_25_adam_le_net5_dense_1_kernel_m3
/assignvariableop_26_adam_le_net5_dense_1_bias_m5
1assignvariableop_27_adam_le_net5_dense_2_kernel_m3
/assignvariableop_28_adam_le_net5_dense_2_bias_m4
0assignvariableop_29_adam_le_net5_conv2d_kernel_v2
.assignvariableop_30_adam_le_net5_conv2d_bias_v6
2assignvariableop_31_adam_le_net5_conv2d_1_kernel_v4
0assignvariableop_32_adam_le_net5_conv2d_1_bias_v3
/assignvariableop_33_adam_le_net5_dense_kernel_v1
-assignvariableop_34_adam_le_net5_dense_bias_v5
1assignvariableop_35_adam_le_net5_dense_1_kernel_v3
/assignvariableop_36_adam_le_net5_dense_1_bias_v5
1assignvariableop_37_adam_le_net5_dense_2_kernel_v3
/assignvariableop_38_adam_le_net5_dense_2_bias_v
identity_40¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ê
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*ö
valueìBé(B$c1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c1/bias/.ATTRIBUTES/VARIABLE_VALUEB$c2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c2/bias/.ATTRIBUTES/VARIABLE_VALUEB$f1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"f1/bias/.ATTRIBUTES/VARIABLE_VALUEB$f2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"f2/bias/.ATTRIBUTES/VARIABLE_VALUEB$f3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"f3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB@c1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>c1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@c2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>c2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@f1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>f1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@f2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>f2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@f3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>f3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@c1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>c1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@c2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>c2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@f1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>f1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@f2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>f2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@f3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>f3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÞ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesö
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¶
_output_shapes£
 ::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¥
AssignVariableOpAssignVariableOp&assignvariableop_le_net5_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1«
AssignVariableOp_1AssignVariableOp&assignvariableop_1_le_net5_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¯
AssignVariableOp_2AssignVariableOp*assignvariableop_2_le_net5_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3­
AssignVariableOp_3AssignVariableOp(assignvariableop_3_le_net5_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¬
AssignVariableOp_4AssignVariableOp'assignvariableop_4_le_net5_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ª
AssignVariableOp_5AssignVariableOp%assignvariableop_5_le_net5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6®
AssignVariableOp_6AssignVariableOp)assignvariableop_6_le_net5_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¬
AssignVariableOp_7AssignVariableOp'assignvariableop_7_le_net5_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8®
AssignVariableOp_8AssignVariableOp)assignvariableop_8_le_net5_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¬
AssignVariableOp_9AssignVariableOp'assignvariableop_9_le_net5_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10¥
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11§
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12§
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¦
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14®
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¡
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¡
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17£
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18£
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¸
AssignVariableOp_19AssignVariableOp0assignvariableop_19_adam_le_net5_conv2d_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¶
AssignVariableOp_20AssignVariableOp.assignvariableop_20_adam_le_net5_conv2d_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21º
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_le_net5_conv2d_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¸
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_le_net5_conv2d_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23·
AssignVariableOp_23AssignVariableOp/assignvariableop_23_adam_le_net5_dense_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24µ
AssignVariableOp_24AssignVariableOp-assignvariableop_24_adam_le_net5_dense_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¹
AssignVariableOp_25AssignVariableOp1assignvariableop_25_adam_le_net5_dense_1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp/assignvariableop_26_adam_le_net5_dense_1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¹
AssignVariableOp_27AssignVariableOp1assignvariableop_27_adam_le_net5_dense_2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28·
AssignVariableOp_28AssignVariableOp/assignvariableop_28_adam_le_net5_dense_2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¸
AssignVariableOp_29AssignVariableOp0assignvariableop_29_adam_le_net5_conv2d_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¶
AssignVariableOp_30AssignVariableOp.assignvariableop_30_adam_le_net5_conv2d_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31º
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adam_le_net5_conv2d_1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¸
AssignVariableOp_32AssignVariableOp0assignvariableop_32_adam_le_net5_conv2d_1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33·
AssignVariableOp_33AssignVariableOp/assignvariableop_33_adam_le_net5_dense_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34µ
AssignVariableOp_34AssignVariableOp-assignvariableop_34_adam_le_net5_dense_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¹
AssignVariableOp_35AssignVariableOp1assignvariableop_35_adam_le_net5_dense_1_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36·
AssignVariableOp_36AssignVariableOp/assignvariableop_36_adam_le_net5_dense_1_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37¹
AssignVariableOp_37AssignVariableOp1assignvariableop_37_adam_le_net5_dense_2_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38·
AssignVariableOp_38AssignVariableOp/assignvariableop_38_adam_le_net5_dense_2_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¸
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39«
Identity_40IdentityIdentity_39:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_40"#
identity_40Identity_40:output:0*³
_input_shapes¡
: :::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ï
õ
'__inference_le_net5_layer_call_fn_38863
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_le_net5_layer_call_and_return_conditional_losses_388372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ  ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1
ª
¨
@__inference_dense_layer_call_and_return_conditional_losses_38766

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	x*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
ñ
#__inference_signature_wrapper_38898
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_386572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ  ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1
©
ª
B__inference_dense_1_layer_call_and_return_conditional_losses_38980

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿx:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
Ç1
É
 __inference__wrapped_model_38657
input_11
-le_net5_conv2d_conv2d_readvariableop_resource2
.le_net5_conv2d_biasadd_readvariableop_resource3
/le_net5_conv2d_1_conv2d_readvariableop_resource4
0le_net5_conv2d_1_biasadd_readvariableop_resource0
,le_net5_dense_matmul_readvariableop_resource1
-le_net5_dense_biasadd_readvariableop_resource2
.le_net5_dense_1_matmul_readvariableop_resource3
/le_net5_dense_1_biasadd_readvariableop_resource2
.le_net5_dense_2_matmul_readvariableop_resource3
/le_net5_dense_2_biasadd_readvariableop_resource
identityÂ
$le_net5/conv2d/Conv2D/ReadVariableOpReadVariableOp-le_net5_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$le_net5/conv2d/Conv2D/ReadVariableOpÒ
le_net5/conv2d/Conv2DConv2Dinput_1,le_net5/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
le_net5/conv2d/Conv2D¹
%le_net5/conv2d/BiasAdd/ReadVariableOpReadVariableOp.le_net5_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%le_net5/conv2d/BiasAdd/ReadVariableOpÄ
le_net5/conv2d/BiasAddBiasAddle_net5/conv2d/Conv2D:output:0-le_net5/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
le_net5/conv2d/BiasAdd
le_net5/conv2d/SigmoidSigmoidle_net5/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
le_net5/conv2d/SigmoidÒ
le_net5/max_pooling2d/MaxPoolMaxPoolle_net5/conv2d/Sigmoid:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
le_net5/max_pooling2d/MaxPoolÈ
&le_net5/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/le_net5_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&le_net5/conv2d_1/Conv2D/ReadVariableOp÷
le_net5/conv2d_1/Conv2DConv2D&le_net5/max_pooling2d/MaxPool:output:0.le_net5/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
paddingVALID*
strides
2
le_net5/conv2d_1/Conv2D¿
'le_net5/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0le_net5_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'le_net5/conv2d_1/BiasAdd/ReadVariableOpÌ
le_net5/conv2d_1/BiasAddBiasAdd le_net5/conv2d_1/Conv2D:output:0/le_net5/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
le_net5/conv2d_1/BiasAdd
le_net5/conv2d_1/SigmoidSigmoid!le_net5/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
le_net5/conv2d_1/SigmoidØ
le_net5/max_pooling2d_1/MaxPoolMaxPoolle_net5/conv2d_1/Sigmoid:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2!
le_net5/max_pooling2d_1/MaxPool
le_net5/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
le_net5/flatten/Constº
le_net5/flatten/ReshapeReshape(le_net5/max_pooling2d_1/MaxPool:output:0le_net5/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
le_net5/flatten/Reshape¸
#le_net5/dense/MatMul/ReadVariableOpReadVariableOp,le_net5_dense_matmul_readvariableop_resource*
_output_shapes
:	x*
dtype02%
#le_net5/dense/MatMul/ReadVariableOp·
le_net5/dense/MatMulMatMul le_net5/flatten/Reshape:output:0+le_net5/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
le_net5/dense/MatMul¶
$le_net5/dense/BiasAdd/ReadVariableOpReadVariableOp-le_net5_dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02&
$le_net5/dense/BiasAdd/ReadVariableOp¹
le_net5/dense/BiasAddBiasAddle_net5/dense/MatMul:product:0,le_net5/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
le_net5/dense/BiasAdd
le_net5/dense/SigmoidSigmoidle_net5/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
le_net5/dense/Sigmoid½
%le_net5/dense_1/MatMul/ReadVariableOpReadVariableOp.le_net5_dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02'
%le_net5/dense_1/MatMul/ReadVariableOp¶
le_net5/dense_1/MatMulMatMulle_net5/dense/Sigmoid:y:0-le_net5/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
le_net5/dense_1/MatMul¼
&le_net5/dense_1/BiasAdd/ReadVariableOpReadVariableOp/le_net5_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02(
&le_net5/dense_1/BiasAdd/ReadVariableOpÁ
le_net5/dense_1/BiasAddBiasAdd le_net5/dense_1/MatMul:product:0.le_net5/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
le_net5/dense_1/BiasAdd
le_net5/dense_1/SigmoidSigmoid le_net5/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
le_net5/dense_1/Sigmoid½
%le_net5/dense_2/MatMul/ReadVariableOpReadVariableOp.le_net5_dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02'
%le_net5/dense_2/MatMul/ReadVariableOp¸
le_net5/dense_2/MatMulMatMulle_net5/dense_1/Sigmoid:y:0-le_net5/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
le_net5/dense_2/MatMul¼
&le_net5/dense_2/BiasAdd/ReadVariableOpReadVariableOp/le_net5_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02(
&le_net5/dense_2/BiasAdd/ReadVariableOpÁ
le_net5/dense_2/BiasAddBiasAdd le_net5/dense_2/MatMul:product:0.le_net5/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
le_net5/dense_2/BiasAdd
le_net5/dense_2/SoftmaxSoftmax le_net5/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
le_net5/dense_2/Softmaxu
IdentityIdentity!le_net5/dense_2/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ  :::::::::::X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1
¯
ª
B__inference_dense_2_layer_call_and_return_conditional_losses_38820

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿT:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
Û
|
'__inference_dense_1_layer_call_fn_38989

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_387932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿx::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
ý
}
(__inference_conv2d_1_layer_call_fn_38938

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_387242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
«
C__inference_conv2d_1_layer_call_and_return_conditional_losses_38929

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2	
Sigmoidg
IdentityIdentitySigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
¨
@__inference_dense_layer_call_and_return_conditional_losses_38960

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	x*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
K
/__inference_max_pooling2d_1_layer_call_fn_38681

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_386752
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
z
%__inference_dense_layer_call_fn_38969

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_387662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*³
serving_default
C
input_18
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ  <
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:æÃ
ò
c1
p1
c2
p2
flatten
f1
f2
f3
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
__call__
_default_save_signature
+&call_and_return_all_conditional_losses"ß
_tf_keras_modelÅ{"class_name": "LeNet5", "name": "le_net5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "LeNet5"}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}}, "metrics": ["sparse_categorical_accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ñ	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
ý
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ö	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 6]}}

regularization_losses
 trainable_variables
!	variables
"	keras_api
__call__
+&call_and_return_all_conditional_losses"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ä
#regularization_losses
$trainable_variables
%	variables
&	keras_api
__call__
+&call_and_return_all_conditional_losses"Ó
_tf_keras_layer¹{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ô

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
__call__
+&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 120, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400]}}
÷

-kernel
.bias
/regularization_losses
0trainable_variables
1	variables
2	keras_api
__call__
+&call_and_return_all_conditional_losses"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 84, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 120]}}
õ

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
__call__
+&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 84}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 84]}}

9iter

:beta_1

;beta_2
	<decay
=learning_ratemvmwmxmy'mz(m{-m|.m}3m~4mvvvv'v(v-v.v3v4v"
	optimizer
 "
trackable_list_wrapper
f
0
1
2
3
'4
(5
-6
.7
38
49"
trackable_list_wrapper
f
0
1
2
3
'4
(5
-6
.7
38
49"
trackable_list_wrapper
Î

regularization_losses
>layer_regularization_losses
?layer_metrics

@layers
Ametrics
trainable_variables
	variables
Bnon_trainable_variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
/:-2le_net5/conv2d/kernel
!:2le_net5/conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
regularization_losses
Clayer_regularization_losses
Dlayer_metrics

Elayers
Fmetrics
trainable_variables
	variables
Gnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
regularization_losses
Hlayer_regularization_losses
Ilayer_metrics

Jlayers
Kmetrics
trainable_variables
	variables
Lnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
1:/2le_net5/conv2d_1/kernel
#:!2le_net5/conv2d_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
regularization_losses
Mlayer_regularization_losses
Nlayer_metrics

Olayers
Pmetrics
trainable_variables
	variables
Qnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
regularization_losses
Rlayer_regularization_losses
Slayer_metrics

Tlayers
Umetrics
 trainable_variables
!	variables
Vnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
#regularization_losses
Wlayer_regularization_losses
Xlayer_metrics

Ylayers
Zmetrics
$trainable_variables
%	variables
[non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
':%	x2le_net5/dense/kernel
 :x2le_net5/dense/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
°
)regularization_losses
\layer_regularization_losses
]layer_metrics

^layers
_metrics
*trainable_variables
+	variables
`non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
(:&xT2le_net5/dense_1/kernel
": T2le_net5/dense_1/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
°
/regularization_losses
alayer_regularization_losses
blayer_metrics

clayers
dmetrics
0trainable_variables
1	variables
enon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
(:&T
2le_net5/dense_2/kernel
": 
2le_net5/dense_2/bias
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
°
5regularization_losses
flayer_regularization_losses
glayer_metrics

hlayers
imetrics
6trainable_variables
7	variables
jnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
»
	mtotal
	ncount
o	variables
p	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
¬
	qtotal
	rcount
s
_fn_kwargs
t	variables
u	keras_api"å
_tf_keras_metricÊ{"class_name": "MeanMetricWrapper", "name": "sparse_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
m0
n1"
trackable_list_wrapper
-
o	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
q0
r1"
trackable_list_wrapper
-
t	variables"
_generic_user_object
4:22Adam/le_net5/conv2d/kernel/m
&:$2Adam/le_net5/conv2d/bias/m
6:42Adam/le_net5/conv2d_1/kernel/m
(:&2Adam/le_net5/conv2d_1/bias/m
,:*	x2Adam/le_net5/dense/kernel/m
%:#x2Adam/le_net5/dense/bias/m
-:+xT2Adam/le_net5/dense_1/kernel/m
':%T2Adam/le_net5/dense_1/bias/m
-:+T
2Adam/le_net5/dense_2/kernel/m
':%
2Adam/le_net5/dense_2/bias/m
4:22Adam/le_net5/conv2d/kernel/v
&:$2Adam/le_net5/conv2d/bias/v
6:42Adam/le_net5/conv2d_1/kernel/v
(:&2Adam/le_net5/conv2d_1/bias/v
,:*	x2Adam/le_net5/dense/kernel/v
%:#x2Adam/le_net5/dense/bias/v
-:+xT2Adam/le_net5/dense_1/kernel/v
':%T2Adam/le_net5/dense_1/bias/v
-:+T
2Adam/le_net5/dense_2/kernel/v
':%
2Adam/le_net5/dense_2/bias/v
ø2õ
'__inference_le_net5_layer_call_fn_38863É
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ  
æ2ã
 __inference__wrapped_model_38657¾
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ  
2
B__inference_le_net5_layer_call_and_return_conditional_losses_38837É
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ  
Ð2Í
&__inference_conv2d_layer_call_fn_38918¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_conv2d_layer_call_and_return_conditional_losses_38909¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
-__inference_max_pooling2d_layer_call_fn_38669à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
°2­
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_38663à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ò2Ï
(__inference_conv2d_1_layer_call_fn_38938¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_conv2d_1_layer_call_and_return_conditional_losses_38929¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
/__inference_max_pooling2d_1_layer_call_fn_38681à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
²2¯
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_38675à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ñ2Î
'__inference_flatten_layer_call_fn_38949¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_flatten_layer_call_and_return_conditional_losses_38944¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_dense_layer_call_fn_38969¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense_layer_call_and_return_conditional_losses_38960¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_1_layer_call_fn_38989¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_1_layer_call_and_return_conditional_losses_38980¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_2_layer_call_fn_39009¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_2_layer_call_and_return_conditional_losses_39000¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2B0
#__inference_signature_wrapper_38898input_1
 __inference__wrapped_model_38657{
'(-.348¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ  
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
³
C__inference_conv2d_1_layer_call_and_return_conditional_losses_38929l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ


 
(__inference_conv2d_1_layer_call_fn_38938_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ

±
A__inference_conv2d_layer_call_and_return_conditional_losses_38909l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
&__inference_conv2d_layer_call_fn_38918_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ¢
B__inference_dense_1_layer_call_and_return_conditional_losses_38980\-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿx
ª "%¢"

0ÿÿÿÿÿÿÿÿÿT
 z
'__inference_dense_1_layer_call_fn_38989O-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿx
ª "ÿÿÿÿÿÿÿÿÿT¢
B__inference_dense_2_layer_call_and_return_conditional_losses_39000\34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿT
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 z
'__inference_dense_2_layer_call_fn_39009O34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿT
ª "ÿÿÿÿÿÿÿÿÿ
¡
@__inference_dense_layer_call_and_return_conditional_losses_38960]'(0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿx
 y
%__inference_dense_layer_call_fn_38969P'(0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿx§
B__inference_flatten_layer_call_and_return_conditional_losses_38944a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
'__inference_flatten_layer_call_fn_38949T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ³
B__inference_le_net5_layer_call_and_return_conditional_losses_38837m
'(-.348¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ  
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
'__inference_le_net5_layer_call_fn_38863`
'(-.348¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ  
ª "ÿÿÿÿÿÿÿÿÿ
í
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_38675R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_1_layer_call_fn_38681R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿë
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_38663R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_max_pooling2d_layer_call_fn_38669R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ®
#__inference_signature_wrapper_38898
'(-.34C¢@
¢ 
9ª6
4
input_1)&
input_1ÿÿÿÿÿÿÿÿÿ  "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
