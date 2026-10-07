add_dataflow_scheduler_library(DataflowSchedulerKTIRToScheduleIR
    ComputeGroupExtraction.cpp
    FuseLinalg.cpp
    IndirectComputeGroupSplit.cpp
    KTIRBufferize.cpp
    KTIRLegalityCheck.cpp
    KTIRMapAndTile.cpp
    KTIRPipeline.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR

  GROUP CONVERSION
  EXPORT_NAME KTIRToScheduleIR

  DEPENDS
    DataflowSchedulerKTIRToScheduleIRPassesIncGen

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    KTIR::SpyreOpDialect
    DataflowScheduler::Analysis
    DataflowScheduler::ArchViews
    DataflowScheduler::KTDFAnalysis
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFArchAnalysis
    DataflowScheduler::KTDFTransforms
    DataflowScheduler::KTDFUtils
    DataflowScheduler::TransformUtils
    DataflowScheduler::Utils
    DataflowScheduler::KTDPLoweringDialect

  LINK_MLIR_LIBS
    MLIRIR
    MLIRAffineDialect
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRTransformUtils
)
