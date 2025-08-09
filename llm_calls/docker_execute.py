import os
import io
import json
import tarfile
import docker
import time

# Correctly import the AnthropicLLM class and the clean_llm_code function
from llm_calls.anthro_llm import AnthropicLLM, clean_llm_code

class DockerScriptRunner:
    """
    Docker-based script runner that uses a single, pre-built image 
    for consistent and fast code execution.
    """
    
    def __init__(self, image_name: str = "data-analyst-agent:latest"):
        """
        Initializes the runner with the name of the pre-built Docker image.
        
        Args:
            image_name (str): The name of the Docker image to use for running scripts.
        """
        self.image_name = image_name
        self._docker_client = None
        print(f"DockerScriptRunner initialized to use image: '{self.image_name}'")
    
    def _get_docker_client(self):
        """Initializes and returns the Docker client, checking for image existence."""
        if self._docker_client is None:
            try:
                self._docker_client = docker.from_env()
                print(f"Checking if Docker image '{self.image_name}' exists...")
                self._docker_client.images.get(self.image_name)
                print("Image found successfully.")
            except docker.errors.ImageNotFound:
                print(f"FATAL ERROR: The required Docker image '{self.image_name}' was not found.")
                print(f"Please build it first by running 'docker build -t {self.image_name} .' in your project directory.")
                raise
            except Exception as e:
                print(f"FATAL ERROR: Could not connect to Docker. Is the Docker daemon running? Error: {e}")
                raise
        return self._docker_client

    def call_python_script(self, script: str, working_dir: str, timeout_sec: int = 120) -> dict:
        """
        Runs a Python script inside a container using the pre-built image.
        
        Args:
            script (str): The Python script to execute.
            working_dir (str): The absolute path to the local directory to mount into the container.
            timeout_sec (int): Timeout for the script execution.
        
        Returns:
            dict: A dictionary containing the execution result (success, stdout, stderr).
        """
        client = self._get_docker_client()
        script = clean_llm_code(script)
        script_name = "script.py"

        container_workdir = "/app"

        container_config = {
            "image": self.image_name,
            "detach": True,
            "working_dir": container_workdir,
            # Add a command to keep the container alive
            "command": "sleep infinity",
            "volumes": {
                os.path.abspath(working_dir): {'bind': container_workdir, 'mode': 'rw'}
            }
        }
        
        container = None
        try:
            # Step 1: Create the container
            container = client.containers.create(**container_config)
            # Step 2: Start the container
            container.start()
            
            # Copy the script into the running container
            script_bytes = script.encode('utf-8')
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                tarinfo = tarfile.TarInfo(name=script_name)
                tarinfo.size = len(script_bytes)
                tar.addfile(tarinfo, io.BytesIO(script_bytes))
            tar_stream.seek(0)
            container.put_archive(path=container_workdir, data=tar_stream)
            
            # Step 3: Execute the script inside the now-running container
            exit_code, (stdout, stderr) = container.exec_run(
                cmd=f"python {script_name}",
                demux=True # demux=True separates stdout and stderr
            )
            
            return {
                "success": exit_code == 0,
                "stdout": stdout.decode("utf-8", errors="replace") if stdout else "",
                "stderr": stderr.decode("utf-8", errors="replace") if stderr else "",
                "script": script
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "script": script}
        finally:
            # Step 4: Stop and remove the container
            if container:
                try:
                    container.remove(force=True)
                except Exception as e:
                    print(f"Warning: Failed to remove container: {e}")

    def run_llm_generated_script(self, system_prompt: str, user_prompt: str, 
                                content=None, working_dir: str = None) -> dict:
        """
        Generates a script using an LLM and runs it in the pre-built Docker container.
        """
        try:
            print("Generating script using Anthropic LLM...")
            # Instantiate the client to call the method
            llm_client = AnthropicLLM()
            generated_script = llm_client.call_anthropic_llm(system_prompt, user_prompt, content)
            print("Script generated successfully.")
            print("-" * 50)
            print(generated_script)
            print("-" * 50)
            
            if not working_dir:
                raise ValueError("working_dir must be provided to run the script in Docker.")

            print("Running script in pre-built Docker container...")
            execution_result = self.call_python_script(
                script=generated_script,
                working_dir=working_dir
            )
            
            return {
                "llm_response": generated_script,
                "execution_result": execution_result,
                "success": execution_result.get("success", False)
            }
            
        except Exception as e:
            return {"error": f"Failed to generate or execute script: {str(e)}", "success": False}
