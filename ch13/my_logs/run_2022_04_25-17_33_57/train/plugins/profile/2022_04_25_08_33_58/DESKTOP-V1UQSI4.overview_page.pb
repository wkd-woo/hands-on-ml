?	????9#??????9#??!????9#??	?#????@@?#????@@!?#????@@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????9#??C?i?q???A?D???J??Y??ʡE???*	33333?@2F
Iterator::Model}гY????!?4;?_U@)?c?ZB??1?`o???T@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate'???????!????U @)㥛? ???1?$??.?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??d?`T??!?~???k@)?Zd;??1ao??F?@:Preprocessing2U
Iterator::Model::ParallelMapV2?<,Ԛ???!σzً@)?<,Ԛ???1σzً@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip0?'???!?Z&??-@)???_vO~?1^(B?]??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice ?o_?y?!8`?|??) ?o_?y?18`?|??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?g??s?u?!By?WW???)?g??s?u?1By?WW???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?????K??!?4??@)??_vOf?1a?3)TC??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 33.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t29.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?#????@@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	C?i?q???C?i?q???!C?i?q???      ??!       "      ??!       *      ??!       2	?D???J???D???J??!?D???J??:      ??!       B      ??!       J	??ʡE?????ʡE???!??ʡE???R      ??!       Z	??ʡE?????ʡE???!??ʡE???JCPU_ONLYY?#????@@b Y      Y@qLP?e??V@"?	
host?Your program is HIGHLY input-bound because 33.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t29.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?91.6331% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 