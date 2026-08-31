add_dataflow_scheduler_library(DataflowSchedulerKTDPLoweringDialect
    KTDPLoweringDialect.cpp
    KTDPLoweringOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Dialect/KTDPLowering
    ${DataflowScheduler_BINARY_DIR}/include/dataflow-scheduler/Dialect/KTDPLowering

  GROUP DIALECT
  EXPORT_NAME KTDPLoweringDialect

  DEPENDS
    DataflowSchedulerKTDPLoweringIncGen

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect

  LINK_MLIR_LIBS
    MLIRIR
    MLIRMemRefDialect
)
