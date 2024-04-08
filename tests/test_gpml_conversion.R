library(testthat)
# Testing read_xml functionality:
test_that("read_xml loads GPML file correctly", {
  expect_s3_class(read_xml("../Desktop/2024/576/pathvisio/data/Hs_Cancer_immunotherapy_by_PD-1_blockade_WP4585_123436.gpml"), "xml_document")
})

# test extract_node function:
test_that("extract_node function works correctly", {
  node <- read_xml("<DataNode TextLabel='TestNode' Type='protein' GraphId='G1234' GroupRef='Ref123'/>")
  extracted <- extract_node(node)
  expect_equal(nrow(extracted), 1)
  expect_equal(extracted$Name, "TestNode")
  expect_equal(extracted$Type, "protein")
})

# Test extract_interaction function:
test_that("extract_interaction function works correctly", {
  interaction <- read_xml("<Interaction GraphId='id12ce76b1'><Point GraphRef=''/><Point GraphRef=''/></Interaction>")
  extracted <- extract_interaction(interaction)
  expect_equal(nrow(extracted), 1)
  expect_true(grepl("id12ce76b1", extracted$interaction_id))
})

# run the tests
test_file("../Desktop/2024/576/pathvisio/data/test_script.R")
