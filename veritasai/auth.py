import bcrypt
import hashlib
from database import store_user, verify_user

def hash_password(password):
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed_password):
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def register_user(username, password):
    """Register a new user."""
    if not username or not password:
        return False, "Username and password are required"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    
    # Hash the password
    password_hash = hash_password(password)
    
    # Store user in database
    success = store_user(username, password_hash)
    
    if success:
        return True, "User registered successfully"
    else:
        return False, "Username already exists"

def login_user(username, password):
    """Login a user."""
    if not username or not password:
        return False, "Username and password are required"
    
    # Get user from database
    user = verify_user(username, password)
    
    if user:
        return True, {"user_id": user[0], "username": user[1]}
    else:
        return False, "Invalid username or password"

def create_session_token(user_id, username):
    """Create a simple session token."""
    import secrets
    token = secrets.token_urlsafe(32)
    return f"{user_id}:{username}:{token}"

def validate_session_token(token):
    """Validate a session token and return user info."""
    try:
        parts = token.split(':')
        if len(parts) == 3:
            user_id, username, _ = parts
            return {"user_id": int(user_id), "username": username}
    except:
        pass
    return None 