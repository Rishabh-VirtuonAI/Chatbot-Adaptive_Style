# utils/vector_builder.py

import os
import re
import json
import torch
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def preprocess_kb_file(kb_file_path: str) -> str:
    """
    Preprocess the knowledge base file to ensure valid JSON format.
    - Removes trailing commas before closing braces/brackets
    - Wraps multiple JSON objects in an array if needed
    - Fixes common JSON formatting issues
    """
    with open(kb_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove trailing commas before closing braces/brackets
    content = re.sub(r',\s*([}\]])(?!\s*[{\[])', r'\1', content)
    
    # Fix missing commas between objects
    content = re.sub(r'}\s*{', '},{', content)
    
    # Ensure proper JSON array format
    content = content.strip()
    if not content.startswith('[') and not content.startswith('{'):
        content = '[' + content + ']'
    elif not content.startswith('[') and content.startswith('{'):
        content = '[' + content + ']'
    
    return content

def parse_kb_content(content):
    """Parse knowledge base content with multiple fallback strategies."""
    # Try parsing as a single JSON array first
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass

    # Try to find and parse individual JSON objects
    objects = []
    decoder = json.JSONDecoder()
    pos = 0
    content = content.strip()

    print("coming inside the parse_kb_content function , with content {}".format(content))

    # If not array, wrap in array
    if content.startswith("{"):
        content = "[" + content + "]"

    # Remove trailing commas
    content = re.sub(r",\s*([}\]])", r"\1", content)

    # Fix missing commas between objects (rare but keeps your old safety)
    content = re.sub(r"}\s*{", "},{", content)

    # Strict JSON load
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return [data]
        return data
    except Exception as e:
        raise ValueError(f"❌ Invalid KB JSON format: {e}")
    
    
    # while pos < len(content):
    #     try:
    #         # Skip whitespace
    #         while pos < len(content) and content[pos].isspace():
    #             pos += 1
    #         if pos >= len(content):
    #             break
                
    #         # Try to parse next JSON object
    #         obj, pos = decoder.raw_decode(content, pos)
    #         objects.append(obj)
    #     except json.JSONDecodeError as e:
    #         # If we can't parse at this position, try the next character
    #         pos += 1
    
    # if objects:
    #     print(f"the objects arre {objects}")
    #     return objects
    
    # # If still no luck, try to fix common JSON issues
    # try:
    #     print("13343144***************")
    #     # Remove trailing commas
    #     content = re.sub(r',\s*([}\]])', r'\1', content)
    #     # Fix missing commas between objects
    #     content = re.sub(r'}\s*{', '},{', content)
    #     # Ensure proper array format
    #     if not content.startswith('[') and not content.startswith('{'):
    #         content = '[' + content + ']'
    #     elif content.startswith('{') and not content.startswith('[{'):
    #         content = '[' + content + ']'
        
    #     data = json.loads(content)
    #     print("----------*****************-----------------")
    #     print(f"the data after formatting is {data}")
    #     print("----------*****************-----------------")
    #     return data if isinstance(data, list) else [data]
    # except json.JSONDecodeError as e:
    #     raise ValueError(f"Could not parse knowledge base: {str(e)}")


def build_faiss_index(domain: str, kb_file_path: str):
    """
    Build and save FAISS vector index for the specified domain.
    Uses regex-based chunking to keep symptoms, causes, and solutions together.
    """

    try:
        # --- Step 1: Read file ---
        if not os.path.exists(kb_file_path):
            raise FileNotFoundError(f"Knowledge base file not found: {kb_file_path}")

        with open(kb_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        if not content:
            raise ValueError("Knowledge base file is empty.")

        # --- Step 2: Parse JSON content ---
        try:
            # First try to parse as a single JSON object
            kb_data = parse_kb_content(content)
            print("parsed the kb data in parse kb content method ")
        except json.JSONDecodeError:
            # If that fails, try to parse as newline-delimited JSON
            try:
                kb_data = [json.loads(line) for line in content.split('\n') if line.strip()]
                print("parsed the kb data in other method ")
            except json.JSONDecodeError:
                # If both fail, try to fix common JSON issues and parse
                content = re.sub(r',\s*([}\]])(?!\s*[{\[])', r'\1', content)  # Remove trailing commas
                content = re.sub(r'}\s*{', '},{', content)  # Add missing commas between objects
                if not content.startswith('[') and not content.startswith('{'):
                    content = '[' + content + ']'
                kb_data = json.loads(content)
                if not isinstance(kb_data, list):
                    kb_data = [kb_data]

        # --- Step 3: Process KB data into chunks ---
        chunks = []
        metadatas = []
        print(f"Processing {len(kb_data)} items...")
        print(f"the kb data that we got is {kb_data}")
        

        def create_chunk(component, data, parent=None):
            """Helper function to create a chunk from component data"""
            chunk_parts = []
            
            # Add component info if available
            if component and component != "generic":
                chunk_parts.append(f"Component: {component}")
            
            # Add symptom/description
            if 'symptom' in data:
                chunk_parts.append(f"Symptom: {data['symptom']}")
            
            # Add possible causes
            if 'possible_causes' in data and data['possible_causes']:
                causes = '\n'.join(f"- {cause}" for cause in data['possible_causes'])
                chunk_parts.append(f"Possible Causes:\n{causes}")
            
            # Add tests and diagnostics
            if 'tests_diagnostics' in data and data['tests_diagnostics']:
                tests = '\n'.join(f"- {test}" for test in data['tests_diagnostics'])
                chunk_parts.append(f"Tests & Diagnostics:\n{tests}")
            
            # Add solutions
            if 'solutions' in data and data['solutions']:
                solutions = '\n'.join(f"- {sol}" for sol in data['solutions'])
                chunk_parts.append(f"Solutions:\n{solutions}")
            
            # Add knowledge facts if available
            if 'knowledge_facts' in data and data['knowledge_facts']:
                facts = '\n'.join(f"• {fact}" for fact in data['knowledge_facts'])
                chunk_parts.append(f"Key Facts:\n{facts}")
            
            # Add analogies if available
            if 'analogy' in data:
                chunk_parts.append(f"Note: {data['analogy']}")
            
            # Add description if available
            if 'description' in data:
                chunk_parts.append(f"Description: {data['description']}")
                
            if 'faq' in data:
                for q,a in zip(data['faq'].get('questions',[]), data['faq'].get('answers',[])):
                    chunk_parts.append(f"FAQ Q: {q}\nFAQ A: {a}")
            
            # Add diagnostic_procedure if available
            if 'diagnostic_procedure' in data:
                chunk_parts.append(f"Diagnostic Procedure: {data['diagnostic_procedure']}")
            
            # Add functionality details if available
            if 'functionality_details' in data:
                # Handle functionality_details as dictionary
                if isinstance(data['functionality_details'], dict):
                    for key, value in data['functionality_details'].items():
                        if isinstance(value, str):
                            chunk_parts.append(f"{key.title()}: {value}")
                # Or as list
                elif isinstance(data['functionality_details'], list):
                    details = '\n'.join(f"- {detail}" for detail in data['functionality_details'])
                    chunk_parts.append(f"Functionality Details:\n{details}")
            
            # Add reference bands if available
            if 'reference_bands' in data and isinstance(data['reference_bands'], dict):
                band_info = []
                for band_type, bands in data['reference_bands'].items():
                    if isinstance(bands, dict):
                        band_list = [f"{k}: {v}" for k, v in bands.items()]
                        band_info.append(f"{band_type}: {', '.join(band_list)}")
                if band_info:
                    chunk_parts.append("Reference Bands:\n" + '\n'.join(f"- {info}" for info in band_info))
            

            # Add fault variations if exist (helps backward compatibility)
            if 'fault' in data and isinstance(data['fault'], list):
                fv = "\n".join(f"- {x}" for x in data['fault'])
                chunk_parts.append(f"Fault Variations:\n{fv}")

            # Add pin-level details if present (old generic fallback, not IC-type)
            if 'pin_number' in data:
                chunk_parts.append(f"Pin Number: {data['pin_number']}")
            if 'pin_name' in data:
                chunk_parts.append(f"Pin Name: {data['pin_name']}")


            # Add pins information if available
            if 'pins' in data and isinstance(data['pins'], dict):
                pin_info = []
                for pin, details in data['pins'].items():
                    if isinstance(details, dict):
                        pin_desc = [f"{pin}:"]
                        if 'function' in details:
                            pin_desc.append(f"  Function: {details['function']}")
                        if 'connections' in details:
                            conns = details['connections']
                            if isinstance(conns, list):
                                conns = '; '.join(conns)
                            pin_desc.append(f"  Connections: {conns}")
                        pin_info.append('\n'.join(pin_desc))
                    else:
                        pin_info.append(f"{pin}: {details}")
                if pin_info:
                    chunk_parts.append("Pin Configuration:\n" + '\n'.join(f"- {p}" for p in pin_info))
            
            return "\n\n".join(chunk_parts)

        def process_ic_kb(ic_data):
            """Process IC-based troubleshooting KB"""

            if not isinstance(ic_data, dict):
                return

            ic_name = ic_data.get("ic_name", "")
            ic_code = ic_data.get("ic_code", "")
            ic_desc = ic_data.get("ic_description", "")
            total_pins = ic_data.get("total_pins", "")

            for pin in ic_data.get("pin_details", []):

                pin_number = pin.get("pin_number")
                pin_name = pin.get("pin_name", "")
                function = pin.get("function", "")
                uses = pin.get("uses", "")
                work = pin.get("work", "")

                base_context = (
                    f"IC Name: {ic_name}\n"
                    f"IC Code: {ic_code}\n"
                    f"IC Description: {ic_desc}\n"
                    f"Total Pins: {total_pins}\n\n"
                    f"Pin Number: {pin_number}\n"
                    f"Pin Name: {pin_name}\n"
                    f"Function: {function}\n"
                    f"Uses: {uses}\n"
                    f"Working: {work}\n"
                )

                for fault_group in pin.get("faults", []):

                    faults = fault_group.get("fault", [])
                    causes = fault_group.get("possible_causes", [])
                    procedures = fault_group.get("diagnostic_procedure", [])
                    tests = fault_group.get("diagnostic_tests", [])
                    solutions = fault_group.get("solution", [])

                    # Safety: force everything to list
                    if isinstance(faults, str): faults = [faults]
                    if isinstance(causes, str): causes = [causes]
                    if isinstance(procedures, str): procedures = [procedures]
                    if isinstance(tests, str): tests = [tests]
                    if isinstance(solutions, str): solutions = [solutions]

                    final_chunk = (
                        base_context +
                        "\nFault Variations:\n" +
                        "\n".join(f"- {f}" for f in faults) +
                        "\n\nPossible Causes:\n" +
                        "\n".join(f"- {c}" for c in causes) +
                        "\n\nDiagnostic Procedure:\n" +
                        "\n".join(f"- {p}" for p in procedures) +
                        "\n\nDiagnostic Tests:\n" +
                        "\n".join(f"- {t}" for t in tests) +
                        "\n\nSolution:\n" +
                        "\n".join(f"- {s}" for s in solutions)
                    )

                    chunks.append(final_chunk)

                    metadatas.append({
                        "ic_name": ic_name,
                        "pin_number": pin_number,
                        "pin_name": pin_name,
                        "fault_variations": faults,
                        "language": "hinglish",
                        "kb_type": "ic_pin_fault"
                    })




        def process_component(component_data, parent_component=None):
            """Process a component and its nested items"""
            if not isinstance(component_data, dict):
                return []
                
            component_chunks = []
            component = component_data.get('component', parent_component or 'generic')
            
            # Create chunk for the component itself
            component_chunks.append(create_chunk(component, component_data))
            
            # Process nested items
            for item_type in ['issues', 'issues_by_symptom']:
                if item_type in component_data and isinstance(component_data[item_type], list):
                    for item in component_data[item_type]:
                        if isinstance(item, dict):
                            component_chunks.append(create_chunk(component, item, component_data))

            # Detect IC-style KB (your new structure)
            if "ic_name" in component_data and "pin_details" in component_data:
                ic_name = component_data.get("ic_name", "")
                ic_code = component_data.get("ic_code", "")
                ic_desc = component_data.get("ic_description", "")
                total_pins = component_data.get("total_pins", "")

                # Create main IC chunk
                ic_main_chunk = (
                    f"IC Name: {ic_name}\n"
                    f"IC Code: {ic_code}\n"
                    f"Description: {ic_desc}\n"
                    f"Total Pins: {total_pins}"
                )
                component_chunks.append(ic_main_chunk)

                # Now process each pin
                for pin in component_data.get("pin_details", []):
                    pin_number = pin.get("pin_number")
                    pin_name = pin.get("pin_name", "")
                    function = pin.get("function", "")
                    uses = pin.get("uses", "")
                    work = pin.get("work", "")

                    base_pin_chunk = (
                        f"IC Pin Information:\n"
                        f"Pin Number: {pin_number}\n"
                        f"Pin Name: {pin_name}\n"
                        f"Function: {function}\n"
                        f"Uses: {uses}\n"
                        f"Work: {work}"
                    )

                    # Each pin has faults list (multiple Hinglish variations)
                    for fault_group in pin.get("faults", []):
                        fault_list = fault_group.get("fault", [])
                        if isinstance(fault_list, str):
                            fault_list = [fault_list]

                        cause_list = fault_group.get("possible_causes", [])
                        if isinstance(cause_list, str):
                            cause_list = [cause_list]

                        proc_list = fault_group.get("diagnostic_procedure", [])
                        if isinstance(proc_list, str):
                            proc_list = [proc_list]

                        test_list = fault_group.get("diagnostic_tests", [])
                        if isinstance(test_list, str):
                            test_list = [test_list]

                        sol_list = fault_group.get("solution", [])
                        if isinstance(sol_list, str):
                            sol_list = [sol_list]

                        final_chunk = (
                            f"{base_pin_chunk}\n\n"
                            f"Fault Variations:\n" + "\n".join(f"- {f}" for f in fault_list) + "\n\n"
                            f"Possible Causes:\n" + "\n".join(f"- {c}" for c in cause_list) + "\n\n"
                            f"Diagnostic Procedure:\n" + "\n".join(f"- {p}" for p in proc_list) + "\n\n"
                            f"Diagnostic Tests:\n" + "\n".join(f"- {t}" for t in test_list) + "\n\n"
                            f"Solution:\n" + "\n".join(f"- {s}" for s in sol_list)
                        )

            component_chunks.append(final_chunk)

                     # component = component_data.get('component', parent_component or 'generic')
            
            # # Create chunk for the component itself
            component = component_data.get('component', parent_component or 'generic')
            component_chunks.append(create_chunk(component, component_data))
            print("component chunks are ", component_chunks)
            # Process nested items
            for item_type in ['issues', 'issues_by_symptom']:
                if item_type in component_data and isinstance(component_data[item_type], list):
                    for item in component_data[item_type]:
                        if isinstance(item, dict):
                            component_chunks.append(create_chunk(component, item, component_data))      


            return component_chunks

        # Main processing loop
        chunks = []
        # for item in kb_data:
        #     if isinstance(item, list):
        #         # Handle list of components
        #         for subitem in item:
        #             if isinstance(subitem, dict):
        #                 chunks.extend(process_component(subitem))
        #     elif isinstance(item, dict):
        #         chunks.extend(process_component(item))
        for item in kb_data:
            if isinstance(item, dict) and "ic_name" in item:
                process_ic_kb(item)

        # Remove empty chunks and deduplicate
        chunks = [c for c in chunks if c.strip()]
        unique_chunks = []
        unique_meta = []
        seen = set()
        # for chunk in chunks:
        #     # Use first 200 chars as key for deduplication
        #     key = chunk[:200].strip()
        #     if key not in seen:
        #         seen.add(key)
        #         unique_chunks.append(chunk)
        for chunk, meta in zip(chunks, metadatas):
            key = chunk[:250]  # semantic-safe dedup key
            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)
                unique_meta.append(meta)

        logger.info("Prepared %d unique chunks for FAISS indexing.", len(unique_chunks))

        # --- Step 4: Create embeddings and build FAISS index ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": True}

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        print(f"the text chunks are {unique_chunks}")
        # Build and save FAISS index
        vectorstore = FAISS.from_texts(unique_chunks, embeddings)
        index_dir = os.path.join("domains", domain, "faiss_index")
        os.makedirs(index_dir, exist_ok=True)
        vectorstore.save_local(index_dir)

        logger.info("✅ FAISS index built and saved to %s", index_dir)

    except Exception as e:
        logger.error("❌ Failed to build FAISS index for domain '%s': %s", domain, e, exc_info=True)
        raise
