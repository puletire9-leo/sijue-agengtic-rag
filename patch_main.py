import sys

path = '/app/backend/open_webui/main.py'
with open(path, 'r') as f:
    content = f.read()

# Add import if missing
if '    starlink,' not in content:
    content = content.replace(
        '    calendar,\n)',
        '    calendar,\n    starlink,\n)'
    )
    print('Added starlink import')
else:
    print('starlink import already present')

# Add router registration if missing
if "starlink.router" not in content:
    content = content.replace(
        "app.include_router(calendar.router, prefix='/api/v1/calendars', tags=['calendars'])",
        "app.include_router(calendar.router, prefix='/api/v1/calendars', tags=['calendars'])\napp.include_router(starlink.router, prefix='/api/v1/starlink', tags=['starlink'])"
    )
    print('Added starlink router registration')
else:
    print('starlink router already registered')

with open(path, 'w') as f:
    f.write(content)
print('Patched main.py successfully')
