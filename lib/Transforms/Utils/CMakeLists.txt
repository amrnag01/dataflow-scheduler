add_dataflow_scheduler_library(DataflowSchedulerTransformUtils
    Hoisting.cpp
    IndirectAccessTileNarrowing.cpp
    LoopSimplification.cpp
    PipelineTreeLegalizer.cpp
    RegionClonePrune.cpp
    SCFTilingUtils.cpp
    Utils.cpp

  EXPORT_NAME TransformUtils

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    DataflowScheduler::Analysis
    DataflowScheduler::KTDPLoweringDialect
    DataflowSchedulerDialects::DataflowDialect
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFArchDialect
    DataflowSchedulerDialects::KTDFTransforms

  LINK_MLIR_LIBS PUBLIC
    MLIRAffineAnalysis
    MLIRAffineDialect
    MLIRArithDialect
    MLIRControlFlowInterfaces
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLinalgUtils
    MLIRLoopLikeInterface
    MLIRMemRefDialect
    MLIRPresburger
    MLIRSCFDialect
    MLIRSCFUtils
    MLIRSideEffectInterfaces
    MLIRTensorDialect
    MLIRValueBoundsOpInterface
    MLIRViewLikeInterface
)
