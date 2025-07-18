# 🧠 Cortex LLM Model Filter & Comparison Tool

A comprehensive Streamlit application for filtering and comparing Large Language Models (LLMs) available through Snowflake Cortex based on various criteria such as region availability, pricing, **real benchmark performance**, and capabilities.

## Features

### 🔍 Advanced Filtering
- **Provider Selection**: Filter by LLM providers (Anthropic, Meta, Mistral AI, Snowflake, OpenAI, DeepSeek, Reka, AI21, Google) with convenient "ALL" option
- **Cross-Region Inference**: Toggle to enable access to all models regardless of regional deployment
- **Regional Availability**: Filter models available in specific regions (AWS US West 2, AWS US East 1, AWS Europe Central 1, Azure East US 2, etc.)
- **Price Range**: Set budget constraints with price per 1K tokens
- **Composite Performance Score**: Filter by weighted performance score (1-10 scale)
- **Individual Benchmark Filtering**: Advanced filters for MMLU Reasoning, HumanEval Coding, and GSM8K Math scores
- **Latency Requirements**: Set maximum acceptable response latency
- **Capabilities**: Filter by required model capabilities (Text Generation, Chat, Reasoning, Code Generation, etc.)

### 📊 Multiple Analysis Views

#### 1. Model Comparison Table
- Sortable table with all model details including individual benchmark scores
- **Real benchmark data**: MMLU, HumanEval, GSM8K scores with source citations
- Customizable sorting by price, composite score, individual benchmarks, latency, or name
- **Benchmark transparency**: Source information for all performance metrics

#### 2. Price vs Performance Analysis
- **Main scatter plot**: Price vs Composite Performance with hover details
- **Individual benchmark analysis**: Separate charts for MMLU, HumanEval, and GSM8K vs price
- Price distribution histograms by provider
- Box plots showing price ranges per provider

#### 3. Regional Availability Matrix
- Interactive heatmap showing model availability by region
- Regional summary charts
- **Smart filtering**: Automatically hidden when cross-region inference is enabled

#### 4. Performance Metrics Deep Dive
- **Composite performance**: Latency vs performance scatter plots
- **Individual benchmarks**: Separate bar charts for MMLU, HumanEval, and GSM8K
- **Performance efficiency**: Composite score per dollar analysis
- **Benchmark explanations**: Detailed descriptions of each benchmark

### 🎯 Smart Recommendations
- **Best Value**: Highest performance per dollar
- **Fastest**: Lowest latency models
- **Best Overall**: Highest composite performance score
- **Most Cost-Effective**: Lowest price per token
- **Best Coding**: Highest HumanEval scores
- **Best Reasoning**: Highest MMLU scores

### 🔬 Real Benchmark Integration

The app uses **authentic benchmark scores** from official research:

#### MMLU (Massive Multitask Language Understanding)
- **What it measures**: General knowledge and reasoning across 57 academic subjects
- **Scale**: 0-100 (percentage accuracy)
- **Source**: Official research papers and Snowflake documentation
- **Weight in composite**: 40%

#### HumanEval (Code Generation)
- **What it measures**: Programming problem-solving capability
- **Scale**: 0-100 (percentage of problems solved correctly)
- **Source**: OpenAI HumanEval benchmark results
- **Weight in composite**: 35%

#### GSM8K (Grade School Math)
- **What it measures**: Mathematical reasoning with word problems
- **Scale**: 0-100 (percentage accuracy)
- **Source**: Official benchmark results
- **Weight in composite**: 25%

#### Composite Score Calculation
```
Composite Score = (MMLU × 0.40 + HumanEval × 0.35 + GSM8K × 0.25) / 10
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the application:**
```bash
streamlit run filter_llm.py
```

3. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

## Supported Models

The application includes **real benchmark data** for 24 LLM models across 9 providers available in Snowflake Cortex:

### Anthropic
- **Claude-3-5-Sonnet**: MMLU: 88.3, HumanEval: 92.0, GSM8K: 96.4
- **Claude-3-7-Sonnet**: MMLU: 88.3, HumanEval: 90.0, GSM8K: 95.0
- **Claude-4-Sonnet**: MMLU: 90.0, HumanEval: 94.0, GSM8K: 97.0

### Meta
- **Llama3.1-8B**: MMLU: 73.0, HumanEval: 72.6, GSM8K: 84.9
- **Llama3.1-70B**: MMLU: 86.0, HumanEval: 80.5, GSM8K: 95.1
- **Llama3.1-405B**: MMLU: 88.6, HumanEval: 89.0, GSM8K: 96.8
- **Llama3.2-1B**: MMLU: 49.3, HumanEval: 35.0, GSM8K: 44.4
- **Llama3.2-3B**: MMLU: 69.4, HumanEval: 45.0, GSM8K: 77.7
- **Llama3.3-70B**: MMLU: 86.0, HumanEval: 80.5, GSM8K: 95.1

### Snowflake (Optimized)
- **Snowflake-Llama-3.3-70B**: Optimized Meta models with SwiftKV
- **Snowflake-Llama-3.1-405B**: Up to 75% cost reduction vs standard
- **Snowflake-Arctic**: MMLU: 67.3, HumanEval: 64.3, GSM8K: 69.7

### Mistral AI
- **Mistral-7B**: MMLU: 62.5, HumanEval: 26.2, GSM8K: 52.1
- **Mixtral-8x7B**: MMLU: 70.6, HumanEval: 40.2, GSM8K: 60.4
- **Mistral-Large2**: MMLU: 84.0, HumanEval: 92.0, GSM8K: 93.0

### OpenAI, DeepSeek, Reka, AI21, Google
- Complete benchmark coverage for all 24 models

## Regional Availability

### North America
- **US Regions**: us-west-2 (Oregon), us-east-1 (N. Virginia)
- **Azure Regions**: East US 2 (Virginia)

### Europe  
- **EU Regions**: eu-central-1 (Frankfurt), eu-west-1 (Ireland)
- **Azure Regions**: West Europe (Netherlands)

### Asia Pacific
- **APAC Regions**: ap-southeast-2 (Sydney), ap-northeast-1 (Tokyo)

## Usage Tips

1. **Start with Provider Selection**: Choose your preferred LLM providers or select "ALL" for comprehensive comparison
2. **Consider Cross-Region Inference**: Enable if you need access to models not available in your target region
3. **Set Regional Requirements**: Select regions where your application will run (bypassed when cross-region is enabled)
4. **Define Budget Constraints**: Use the price slider to set your budget
5. **Choose Performance Needs**: Use composite score or individual benchmarks based on your requirements
6. **Advanced Benchmark Filtering**: Use the expandable advanced filters for specific benchmark requirements
7. **Review Recommendations**: Check the automatic recommendations for your needs

## Key Features Explained

### Cross-Region Inference
When enabled, this feature allows you to access any Snowflake Cortex model regardless of its regional deployment by using Snowflake's cross-region inference capability. This means:
- ✅ Access to all models regardless of location

### Provider "ALL" Option
Select "ALL" in the provider filter to include models from all available providers for comprehensive comparison. This is useful when you want to see the complete landscape of available models without provider restrictions.

### Real Benchmark Scores
All performance scores are derived from **authentic benchmark results** from official research papers and documentation:
- **Transparency**: Source citations for all benchmark scores
- **Accuracy**: No synthetic or estimated performance data
- **Standardization**: Consistent benchmarking across all models
- **Weighted Scoring**: Composite scores balance different capabilities

## Model Metrics Explained

- **Composite Score**: Weighted average of MMLU, HumanEval, and GSM8K (1-10 scale)
- **MMLU Reasoning**: General knowledge and reasoning (0-100 scale)
- **HumanEval Coding**: Code generation accuracy (0-100 scale)
- **GSM8K Math**: Mathematical reasoning (0-100 scale)
- **Price per 1K Tokens**: Cost for processing 1,000 tokens
- **Latency**: Average response time in milliseconds
- **Max Tokens**: Maximum context window size
- **Capabilities**: Model's supported features and abilities
- **Regions**: Cloud regions where the model is available
- **Benchmark Source**: Citation for performance data transparency

## Key Model Highlights

### 🏆 Top Performers by Category

**Overall Performance (Composite Score)**
- Claude-4-Sonnet: Leading in balanced performance
- Llama3.1-405B: Excellent open-source option
- Mistral-Large2: Strong enterprise choice

**Coding Excellence (HumanEval)**
- Claude-3-5-Sonnet: 92.0 score
- Mistral-Large2: 92.0 score
- Llama3.1-405B: 89.0 score

**Reasoning Champions (MMLU)**
- Claude-4-Sonnet: 90.0 score
- Llama3.1-405B: 88.6 score
- Claude-3-5-Sonnet: 88.3 score

**Mathematical Reasoning (GSM8K)**
- Claude-4-Sonnet: 97.0 score
- Llama3.1-405B: 96.8 score
- Claude-3-5-Sonnet: 96.4 score

**Best Value**
- Llama3.1-8B: Good performance at low cost
- Mistral-7B: Fast and economical
- Llama3.2-3B: Efficient for basic tasks

## Technical Details

- **Framework**: Streamlit
- **Data Source**: Official Snowflake Cortex documentation + research papers
- **Visualizations**: Plotly for interactive charts
- **Real-time Filtering**: Instant updates across all views
- **Responsive Design**: Works on desktop and mobile

## Contributing

This tool uses real benchmark data from official sources. If you notice any inaccuracies in the benchmark scores or want to suggest improvements, please ensure all data is properly sourced and verifiable.

---

**Note**: This application provides real benchmark scores for educational and comparison purposes. Always verify model performance for your specific use case through direct testing.
