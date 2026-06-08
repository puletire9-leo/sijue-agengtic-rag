"""Incremental patch for Open WebUI official image.
Only adds starlink_kb_ids handling - doesn't replace entire files."""
import re

def patch_middleware():
    path = '/app/backend/open_webui/utils/middleware.py'
    with open(path, 'r') as f:
        content = f.read()
    
    if 'starlink_kb_ids' in content:
        print('middleware.py already patched')
        return
    
    # Find: if files := body.get('metadata', {}).get('files', None):
    # Add starlink extraction before the original file handling
    old = "    if files := body.get('metadata', {}).get('files', None):"
    new = """    if files := body.get('metadata', {}).get('files', None):
        # Separate starlink KBs from regular files
        starlink_kb_ids = []
        regular_files = []
        for item in files:
            if item.get('_is_starlink'):
                starlink_kb_ids.append(item.get('id', ''))
            else:
                regular_files.append(item)
        if starlink_kb_ids:
            body.setdefault('metadata', {})['starlink_kb_ids'] = starlink_kb_ids
        files = regular_files if regular_files else None
        if files is not None:
            body['metadata']['files'] = files

    if files := body.get('metadata', {}).get('files', None):
        pass  # placeholder removed by second patch below
"""
    # We need a different approach - find the exact pattern and inject
    # Look for the line right after "if files := body.get('metadata', {}).get('files', None):"
    # in chat_completion_files_handler function
    
    # Simpler: find the function and inject at the top
    func_marker = "async def chat_completion_files_handler("
    if func_marker not in content:
        print('WARNING: chat_completion_files_handler not found in middleware.py')
        return
    
    # Find the function and the "if files" line
    # We insert our code BEFORE the existing "if files" check
    target = "    if files := body.get('metadata', {}).get('files', None):\n"
    # Count occurrences - we want the one inside chat_completion_files_handler
    idx = content.find(func_marker)
    files_idx = content.find(target, idx)
    
    if files_idx == -1:
        print('WARNING: files check not found in middleware.py')
        return
    
    injection = """    if files := body.get('metadata', {}).get('files', None):
        # Separate starlink KBs from regular files
        starlink_kb_ids = []
        regular_files = []
        for item in files:
            if item.get('_is_starlink'):
                starlink_kb_ids.append(item.get('id', ''))
            else:
                regular_files.append(item)
        if starlink_kb_ids:
            body.setdefault('metadata', {})['starlink_kb_ids'] = starlink_kb_ids
        files = regular_files if regular_files else None
        if files is not None:
            body['metadata']['files'] = files

"""
    content = content[:files_idx] + injection + content[files_idx:]
    
    with open(path, 'w') as f:
        f.write(content)
    print('Patched middleware.py')


def patch_openai():
    path = '/app/backend/open_webui/routers/openai.py'
    with open(path, 'r') as f:
        content = f.read()
    
    if 'starlink_kb_ids' in content:
        print('openai.py already patched')
        return
    
    # Find where payload is sent to the backend
    # We need to add: payload['starlink_kb_ids'] = metadata.get('starlink_kb_ids', [])
    # Right before the fetch_url or response creation
    
    # Look for metadata variable usage near payload construction
    # The key line is typically: payload = { ... } or after payload is built
    
    # Find: "metadata = body.get('metadata', {})" or similar
    # Then inject after payload is constructed
    
    # Simplest: find "payload = {" and inject after the dict is complete
    # Actually, find the response creation and inject before it
    
    # Look for the chat completion function (name varies by version)
    marker = None
    for candidate in ["async def chat_completion(", "async def generate_chat_completion("]:
        if candidate in content:
            marker = candidate
            break
    if not marker:
        print('WARNING: chat completion function not found')
        return
    
    # Find: metadata = payload.pop('metadata', None)
    meta_marker = "metadata = payload.pop('metadata', None)"
    meta_idx = content.find(meta_marker, content.find(marker))

    if meta_idx == -1:
        print('WARNING: metadata extraction not found in openai.py')
        return

    # Find the end of the line
    line_end = content.find('\n', meta_idx)

    injection = """
    # Inject starlink_kb_ids for SuperMew backend
    if metadata and metadata.get('starlink_kb_ids'):
        payload['starlink_kb_ids'] = metadata['starlink_kb_ids']
"""

    content = content[:line_end+1] + injection + content[line_end+1:]
    
    with open(path, 'w') as f:
        f.write(content)
    print('Patched openai.py')


if __name__ == '__main__':
    patch_middleware()
    patch_openai()
