add_dataflow_scheduler_library(DataflowSchedulerUtils
    SchedulerExtContext.cpp
    TileSizeAgentInterface.cpp
    AnthropicAgentClient.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Utils

  EXPORT_NAME Utils

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    DataflowScheduler::Analysis
    DataflowScheduler::ArchViews
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFArchDialect
    DataflowSchedulerDialects::KTDFArchAnalysis

  LINK_MLIR_LIBS
    MLIRIR
)

# Link libcurl and nlohmann_json privately
target_link_libraries(DataflowSchedulerUtils PRIVATE CURL::libcurl nlohmann_json::nlohmann_json)
