add_dataflow_scheduler_library(DataflowSchedulerArchViews
    GroupLocalMemory.cpp
    MemoryTree.cpp
    RoutingGraph.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Analysis/ArchViews

  EXPORT_NAME ArchViews

  LINK_LIBS PUBLIC
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFArchDialect
    DataflowSchedulerDialects::KTDFArchAnalysis
    KTIR::KtdpDialect

  LINK_MLIR_LIBS
    MLIRIR
)
