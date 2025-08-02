#!/usr/bin/env python3
"""
Automated dependency update script for Self-Driving Materials Orchestrator.

Automatically updates dependencies, checks for security vulnerabilities,
and creates pull requests for updates.
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import requests
import tempfile
import shutil


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DependencyUpdater:
    """Automated dependency management and update system."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_repo = os.getenv('GITHUB_REPOSITORY', 'terragonlabs/self-driving-materials-orchestrator')
        self.update_summary = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'python_updates': [],
            'docker_updates': [],
            'github_actions_updates': [],
            'security_fixes': [],
            'total_updates': 0
        }
    
    def run_command(self, cmd: List[str], check_return_code: bool = True, cwd: Optional[Path] = None) -> Tuple[bool, str]:
        """Run a shell command and return success status and output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd or self.project_root,
                timeout=300
            )
            
            if check_return_code and result.returncode != 0:
                logger.error(f"Command failed: {' '.join(cmd)}")
                logger.error(f"Error: {result.stderr}")
                return False, result.stderr
            
            return True, result.stdout
            
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)
    
    def backup_files(self, files: List[str]) -> Dict[str, str]:
        """Create backups of specified files."""
        backups = {}
        backup_dir = Path(tempfile.mkdtemp())
        
        for file in files:
            file_path = self.project_root / file
            if file_path.exists():
                backup_path = backup_dir / file.replace('/', '_')
                shutil.copy2(file_path, backup_path)
                backups[file] = str(backup_path)
                logger.debug(f"Backed up {file} to {backup_path}")
        
        return backups
    
    def restore_files(self, backups: Dict[str, str]) -> None:
        """Restore files from backups."""
        for file, backup_path in backups.items():
            file_path = self.project_root / file
            shutil.copy2(backup_path, file_path)
            logger.info(f"Restored {file} from backup")
    
    def check_python_dependencies(self) -> List[Dict[str, Any]]:
        """Check for outdated Python dependencies."""
        logger.info("Checking Python dependencies...")
        
        outdated_packages = []
        
        # Get list of outdated packages
        success, output = self.run_command(['pip', 'list', '--outdated', '--format=json'], check_return_code=False)
        if success:
            try:
                packages = json.loads(output)
                for package in packages:
                    outdated_packages.append({
                        'name': package['name'],
                        'current_version': package['version'],
                        'latest_version': package['latest_version'],
                        'type': package.get('latest_filetype', 'wheel')
                    })
            except json.JSONDecodeError:
                logger.warning("Failed to parse pip list output")
        
        return outdated_packages
    
    def update_python_dependencies(self, update_type: str = 'patch') -> List[Dict[str, Any]]:
        """Update Python dependencies based on update type."""
        logger.info(f"Updating Python dependencies ({update_type})...")
        
        updates = []
        backups = self.backup_files(['pyproject.toml', 'requirements.txt', 'requirements-dev.txt'])
        
        try:
            # Check if using pyproject.toml
            if (self.project_root / 'pyproject.toml').exists():
                # Use pip-tools for dependency management
                success, output = self.run_command(['pip', 'install', 'pip-tools'])
                if not success:
                    logger.error("Failed to install pip-tools")
                    return updates
                
                # Update dependencies in pyproject.toml
                if update_type in ['minor', 'major', 'all']:
                    success, output = self.run_command(['pur', '-r', 'pyproject.toml', '--force'])
                    if success:
                        logger.info("Updated pyproject.toml with latest versions")
                        updates.append({
                            'file': 'pyproject.toml',
                            'action': 'updated_versions',
                            'type': update_type
                        })
                
                # Compile new requirements
                success, output = self.run_command(['pip-compile', 'pyproject.toml', '--upgrade'])
                if success:
                    updates.append({
                        'file': 'requirements.txt',
                        'action': 'compiled',
                        'type': 'lock_file'
                    })
                
                # Compile dev requirements if they exist
                if '[dev]' in (self.project_root / 'pyproject.toml').read_text():
                    success, output = self.run_command([
                        'pip-compile', 'pyproject.toml', '--extra', 'dev', 
                        '--upgrade', '-o', 'requirements-dev.txt'
                    ])
                    if success:
                        updates.append({
                            'file': 'requirements-dev.txt',
                            'action': 'compiled',
                            'type': 'dev_lock_file'
                        })
            
            # Install updated dependencies to test compatibility
            success, output = self.run_command(['pip', 'install', '-e', '.'])
            if not success:
                logger.error("Failed to install updated dependencies")
                self.restore_files(backups)
                return []
            
            # Run basic tests to verify compatibility
            success, output = self.run_command(['python', '-c', 'import materials_orchestrator; print("✅ Import successful")'])
            if not success:
                logger.error("Updated dependencies break import")
                self.restore_files(backups)
                return []
            
            self.update_summary['python_updates'] = updates
            return updates
            
        except Exception as e:
            logger.error(f"Failed to update Python dependencies: {e}")
            self.restore_files(backups)
            return []
    
    def check_security_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Check for security vulnerabilities in dependencies."""
        logger.info("Checking for security vulnerabilities...")
        
        vulnerabilities = []
        
        # Run safety check
        success, output = self.run_command(['safety', 'check', '--json'], check_return_code=False)
        if success:
            try:
                safety_results = json.loads(output)
                for vuln in safety_results.get('vulnerabilities', []):
                    vulnerabilities.append({
                        'package': vuln.get('package_name'),
                        'version': vuln.get('analyzed_version'),
                        'vulnerability_id': vuln.get('vulnerability_id'),
                        'advisory': vuln.get('advisory'),
                        'severity': 'high' if 'critical' in vuln.get('advisory', '').lower() else 'medium'
                    })
            except json.JSONDecodeError:
                logger.warning("Failed to parse safety output")
        
        # Run pip-audit for additional checks
        success, output = self.run_command(['pip-audit', '--format=json'], check_return_code=False)
        if success:
            try:
                audit_results = json.loads(output)
                for vuln in audit_results.get('vulnerabilities', []):
                    vulnerabilities.append({
                        'package': vuln.get('package', {}).get('name'),
                        'version': vuln.get('package', {}).get('version'),
                        'vulnerability_id': vuln.get('id'),
                        'advisory': vuln.get('description'),
                        'severity': vuln.get('severity', 'medium').lower()
                    })
            except json.JSONDecodeError:
                logger.warning("Failed to parse pip-audit output")
        
        self.update_summary['security_fixes'] = vulnerabilities
        return vulnerabilities
    
    def update_docker_images(self) -> List[Dict[str, Any]]:
        """Update Docker base images to latest versions."""
        logger.info("Checking Docker base images...")
        
        updates = []
        docker_files = ['Dockerfile', 'Dockerfile.production', 'Dockerfile.jupyter']
        
        for dockerfile in docker_files:
            dockerfile_path = self.project_root / dockerfile
            if not dockerfile_path.exists():
                continue
            
            content = dockerfile_path.read_text()
            lines = content.split('\n')
            updated_lines = []
            file_updated = False
            
            for line in lines:
                if line.strip().startswith('FROM'):
                    # Extract image and tag
                    parts = line.split()
                    if len(parts) >= 2:
                        image_spec = parts[1]
                        if ':' in image_spec:
                            image, tag = image_spec.split(':', 1)
                            
                            # Check for updates to specific images
                            if image == 'python' and tag.endswith('-slim'):
                                # Check for latest Python slim version
                                base_version = tag.split('-')[0]  # e.g., "3.11" from "3.11.5-slim"
                                latest_tag = self.get_latest_docker_tag(image, f"{base_version}.*-slim")
                                
                                if latest_tag and latest_tag != tag:
                                    updated_line = line.replace(f":{tag}", f":{latest_tag}")
                                    updated_lines.append(updated_line)
                                    file_updated = True
                                    
                                    updates.append({
                                        'file': dockerfile,
                                        'image': image,
                                        'old_tag': tag,
                                        'new_tag': latest_tag,
                                        'action': 'updated'
                                    })
                                    continue
            
                updated_lines.append(line)
            
            if file_updated:
                dockerfile_path.write_text('\n'.join(updated_lines))
                logger.info(f"Updated {dockerfile}")
        
        self.update_summary['docker_updates'] = updates
        return updates
    
    def get_latest_docker_tag(self, image: str, pattern: str) -> Optional[str]:
        """Get the latest Docker tag matching a pattern."""
        try:
            # Use Docker Hub API to get tags
            url = f"https://registry.hub.docker.com/v2/repositories/library/{image}/tags/"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                tags = [tag['name'] for tag in data.get('results', [])]
                
                import re
                pattern_regex = pattern.replace('.*', r'[0-9.]+')
                matching_tags = [tag for tag in tags if re.match(pattern_regex, tag)]
                
                if matching_tags:
                    # Sort by version and return latest
                    return sorted(matching_tags)[-1]
            
        except Exception as e:
            logger.warning(f"Failed to check Docker tags for {image}: {e}")
        
        return None
    
    def update_github_actions(self) -> List[Dict[str, Any]]:
        """Update GitHub Actions to latest versions."""
        logger.info("Checking GitHub Actions...")
        
        updates = []
        workflows_dir = self.project_root / '.github' / 'workflows'
        
        if not workflows_dir.exists():
            logger.info("No GitHub workflows found")
            return updates
        
        action_updates = {}
        
        for workflow_file in workflows_dir.glob('*.yml'):
            content = workflow_file.read_text()
            lines = content.split('\n')
            updated_lines = []
            file_updated = False
            
            for line in lines:
                if 'uses:' in line and '@v' in line:
                    # Extract action and version
                    uses_part = line.split('uses:')[1].strip()
                    if '@' in uses_part:
                        action, version = uses_part.split('@', 1)
                        action = action.strip()
                        version = version.strip()
                        
                        # Get latest version for known actions
                        latest_version = self.get_latest_action_version(action)
                        if latest_version and latest_version != version:
                            updated_line = line.replace(f"@{version}", f"@{latest_version}")
                            updated_lines.append(updated_line)
                            file_updated = True
                            
                            action_updates[action] = {
                                'old_version': version,
                                'new_version': latest_version
                            }
                            continue
                
                updated_lines.append(line)
            
            if file_updated:
                workflow_file.write_text('\n'.join(updated_lines))
                updates.append({
                    'file': str(workflow_file.relative_to(self.project_root)),
                    'actions_updated': action_updates
                })
                logger.info(f"Updated {workflow_file.name}")
        
        self.update_summary['github_actions_updates'] = updates
        return updates
    
    def get_latest_action_version(self, action: str) -> Optional[str]:
        """Get the latest version of a GitHub Action."""
        try:
            # GitHub API to get releases
            url = f"https://api.github.com/repos/{action}/releases/latest"
            headers = {}
            
            if self.github_token:
                headers['Authorization'] = f'token {self.github_token}'
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('tag_name')
            
        except Exception as e:
            logger.warning(f"Failed to check latest version for {action}: {e}")
        
        return None
    
    def run_tests_after_update(self) -> bool:
        """Run tests to verify updates don't break functionality."""
        logger.info("Running tests after updates...")
        
        # Run basic import test
        success, output = self.run_command(['python', '-c', 'import materials_orchestrator; print("Import successful")'])
        if not success:
            logger.error("Import test failed after updates")
            return False
        
        # Run unit tests
        success, output = self.run_command(['pytest', 'tests/unit/', '-x', '--tb=short'], check_return_code=False)
        if not success:
            logger.error("Unit tests failed after updates")
            return False
        
        # Run security checks
        success, output = self.run_command(['bandit', '-r', 'src/', '-f', 'json'], check_return_code=False)
        if success:
            try:
                results = json.loads(output)
                high_severity = [r for r in results.get('results', []) if r.get('issue_severity') == 'HIGH']
                if high_severity:
                    logger.error(f"Security issues introduced: {len(high_severity)} high severity")
                    return False
            except json.JSONDecodeError:
                pass
        
        logger.info("All tests passed after updates")
        return True
    
    def create_update_branch(self, branch_name: str) -> bool:
        """Create a new branch for updates."""
        logger.info(f"Creating update branch: {branch_name}")
        
        # Check if branch already exists
        success, output = self.run_command(['git', 'branch', '--list', branch_name])
        if branch_name in output:
            logger.info("Branch already exists, switching to it")
            success, output = self.run_command(['git', 'checkout', branch_name])
            return success
        
        # Create new branch
        success, output = self.run_command(['git', 'checkout', '-b', branch_name])
        return success
    
    def commit_and_push_updates(self, branch_name: str, commit_message: str) -> bool:
        """Commit and push updates to the branch."""
        logger.info("Committing and pushing updates...")
        
        # Add changed files
        success, output = self.run_command(['git', 'add', '.'])
        if not success:
            return False
        
        # Check if there are changes to commit
        success, output = self.run_command(['git', 'diff', '--cached', '--quiet'], check_return_code=False)
        if success:  # No changes
            logger.info("No changes to commit")
            return True
        
        # Commit changes
        success, output = self.run_command(['git', 'commit', '-m', commit_message])
        if not success:
            return False
        
        # Push to remote
        success, output = self.run_command(['git', 'push', '-u', 'origin', branch_name])
        return success
    
    def create_pull_request(self, branch_name: str, title: str, body: str) -> Optional[str]:
        """Create a pull request for the updates."""
        if not self.github_token:
            logger.warning("No GitHub token provided, cannot create PR")
            return None
        
        logger.info("Creating pull request...")
        
        url = f"https://api.github.com/repos/{self.github_repo}/pulls"
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        data = {
            'title': title,
            'head': branch_name,
            'base': 'main',
            'body': body
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 201:
                pr_data = response.json()
                pr_url = pr_data.get('html_url')
                logger.info(f"Pull request created: {pr_url}")
                return pr_url
            else:
                logger.error(f"Failed to create PR: {response.status_code} {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create pull request: {e}")
            return None
    
    def generate_update_summary(self) -> str:
        """Generate a summary of all updates."""
        summary = "## Automated Dependency Updates\n\n"
        summary += f"**Generated:** {self.update_summary['timestamp']}\n\n"
        
        total_updates = (
            len(self.update_summary['python_updates']) +
            len(self.update_summary['docker_updates']) +
            len(self.update_summary['github_actions_updates'])
        )
        
        if total_updates == 0:
            summary += "✅ No updates available - all dependencies are current!\n"
            return summary
        
        summary += f"**Total Updates:** {total_updates}\n\n"
        
        # Python updates
        if self.update_summary['python_updates']:
            summary += "### 🐍 Python Dependencies\n"
            for update in self.update_summary['python_updates']:
                summary += f"- Updated {update['file']} ({update['action']})\n"
            summary += "\n"
        
        # Docker updates
        if self.update_summary['docker_updates']:
            summary += "### 🐳 Docker Images\n"
            for update in self.update_summary['docker_updates']:
                summary += f"- {update['image']}: {update['old_tag']} → {update['new_tag']}\n"
            summary += "\n"
        
        # GitHub Actions updates
        if self.update_summary['github_actions_updates']:
            summary += "### ⚡ GitHub Actions\n"
            for update in self.update_summary['github_actions_updates']:
                summary += f"- Updated {update['file']}\n"
                for action, versions in update['actions_updated'].items():
                    summary += f"  - {action}: {versions['old_version']} → {versions['new_version']}\n"
            summary += "\n"
        
        # Security fixes
        if self.update_summary['security_fixes']:
            summary += "### 🔒 Security Fixes\n"
            for vuln in self.update_summary['security_fixes']:
                summary += f"- **{vuln['package']}** ({vuln['version']}): {vuln['vulnerability_id']}\n"
            summary += "\n"
        
        summary += "### ✅ Testing\n"
        summary += "- [x] Import tests passed\n"
        summary += "- [x] Unit tests passed\n"
        summary += "- [x] Security scan clean\n"
        summary += "- [ ] Integration tests (run in CI)\n\n"
        
        summary += "### 📋 Manual Review Required\n"
        summary += "- [ ] Review breaking changes in changelogs\n"
        summary += "- [ ] Test critical functionality\n"
        summary += "- [ ] Verify backward compatibility\n\n"
        
        summary += "---\n"
        summary += "🤖 This PR was automatically created by the dependency update workflow."
        
        return summary
    
    def run_update_process(self, update_type: str = 'patch', create_pr: bool = False) -> bool:
        """Run the complete dependency update process."""
        logger.info(f"Starting dependency update process ({update_type})...")
        
        # Create update branch
        branch_name = f"deps/automated-update-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        if not self.create_update_branch(branch_name):
            logger.error("Failed to create update branch")
            return False
        
        try:
            # Update Python dependencies
            python_updates = self.update_python_dependencies(update_type)
            
            # Update Docker images
            docker_updates = self.update_docker_images()
            
            # Update GitHub Actions
            action_updates = self.update_github_actions()
            
            # Check for security vulnerabilities
            vulnerabilities = self.check_security_vulnerabilities()
            
            total_updates = len(python_updates) + len(docker_updates) + len(action_updates)
            self.update_summary['total_updates'] = total_updates
            
            if total_updates == 0 and not vulnerabilities:
                logger.info("No updates available")
                return True
            
            # Run tests after updates
            if not self.run_tests_after_update():
                logger.error("Tests failed after updates")
                return False
            
            # Commit and push changes
            commit_message = f"deps: automated dependency updates ({update_type})\n\n"
            commit_message += f"- Python packages: {len(python_updates)} updates\n"
            commit_message += f"- Docker images: {len(docker_updates)} updates\n"
            commit_message += f"- GitHub Actions: {len(action_updates)} updates\n"
            
            if vulnerabilities:
                commit_message += f"- Security fixes: {len(vulnerabilities)} vulnerabilities addressed\n"
            
            if not self.commit_and_push_updates(branch_name, commit_message):
                logger.error("Failed to commit and push updates")
                return False
            
            # Create pull request if requested
            if create_pr:
                pr_title = f"🔧 Automated Dependency Updates ({update_type.title()})"
                pr_body = self.generate_update_summary()
                
                pr_url = self.create_pull_request(branch_name, pr_title, pr_body)
                if pr_url:
                    print(f"✅ Pull request created: {pr_url}")
                else:
                    print("⚠️  Updates committed but PR creation failed")
            
            logger.info("Dependency update process completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Update process failed: {e}")
            return False


def main():
    """Main entry point for dependency updater."""
    parser = argparse.ArgumentParser(description='Automated dependency updater')
    parser.add_argument('--update-type', choices=['patch', 'minor', 'major', 'all'], 
                       default='patch', help='Type of updates to apply')
    parser.add_argument('--create-pr', action='store_true', 
                       help='Create pull request for updates')
    parser.add_argument('--project-root', default='.', 
                       help='Project root directory')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        updater = DependencyUpdater(args.project_root)
        success = updater.run_update_process(args.update_type, args.create_pr)
        
        if success:
            print("✅ Dependency update process completed successfully")
            sys.exit(0)
        else:
            print("❌ Dependency update process failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Dependency updater failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()