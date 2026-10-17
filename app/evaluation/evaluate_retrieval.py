VectorTechnics Catalogue 2026 — RAG Testing Report 

1. Document Overview 

Parameter 

Details 

Document 

VectorTechnics Catalogue 2026 

File size 

128.66 MB 

Pages 

114 

Document type 

Technical product catalogue 

Domain 

UAV propulsion and power electronics 

Previous test document 

ideaForge Annual Report 

Objective: Evaluate the document-independent RAG pipeline on a technical catalogue containing structured numerical data, engineering specifications, performance tables, and diagrams. 

2. Types of Data Present 

The document contains five major types of information: 

Data type 

Examples 

Technical specifications 

Motor KV, torque, voltage, power, weight 

Experimental data 

RPM, thrust, current, efficiency at different throttle levels 

Structured tables 

Product specifications and performance measurements 

Visual information 

Motor diagrams, 2D CAD drawings, product images 

Descriptive information 

Manufacturing capabilities, propulsion architecture, product applications 

The catalogue covers BLDC motors, VTOL systems, propellers, ESCs, power distribution boards, and DC-DC converters. 

3. Comparison with ideaForge 

Feature 

ideaForge 

VectorTechnics 

Document nature 

Annual report 

Technical catalogue 

Primary information 

Financial and business 

Engineering and numerical 

Text structure 

Narrative-heavy 

Table-heavy 

Numerical complexity 

Moderate 

High 

Visual complexity 

Moderate 

High 

Main retrieval challenge 

Finding financial disclosures 

Matching exact products, values, and conditions 

4. Expected Impact on RAG Pipeline 

Pipeline stage 

Expected challenge 

Ingestion 

Extracting complex tables and engineering symbols 

Cleaning 

Preserving units, numerical values, and product identifiers 

Recursive chunking 

Risk of splitting table rows and specifications 

Structure-aware chunking 

Preserving product-specific tables and metadata 

Contextual chunking 

Associating measurements with correct product models 

Embeddings 

Distinguishing products with similar technical specifications 

Retrieval and reranking 

Identifying exact values under specific test conditions 

LLM generation 

Preventing incorrect numbers, units, and product comparisons 

5. Proposed Evaluation 

The existing three chunking strategies will be tested using approximately 10 technical questions. 

Question categories: factual retrieval, numerical extraction, product comparison, table interpretation, multi-row reasoning, and technical understanding. 

Evaluation metrics: 

Hit@1, Hit@3, Hit@5, Hit@10 

MRR@5 and MRR@10 

Recall@5 with verified relevance labels 

Answer correctness and faithfulness 

Numerical and unit accuracy 

Chunking time and embedding time 

End-to-end response latency 

All strategies will use identical questions and evaluation criteria. 

6. Expected Outcomes 

The experiment will help determine: 

Which chunking strategy best preserves technical information. 

Whether contextual information improves product-specific retrieval. 

How effectively the pipeline handles complex tables and numerical measurements. 

Whether the LLM can generate accurate, evidence-grounded technical answers. 

How document structure affects processing time, retrieval accuracy, and answer quality. 

These are evaluation objectives; actual performance has not yet been measured. 

7. Conclusion 

The VectorTechnics catalogue provides a substantially different testing environment compared with the ideaForge annual report. 

While ideaForge primarily evaluates retrieval from financial and narrative content, VectorTechnics evaluates technical precision, structured-table retrieval, numerical reasoning, and product-specific context preservation. 

Testing both documents will provide stronger evidence of the RAG pipeline's document independence, robustness, and retrieval effectiveness across different data structures. 

Current status: Document reviewed; new PDF ingestion and benchmark results are still pending verification. 

 