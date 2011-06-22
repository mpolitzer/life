#include <string.h>
#include <ctype.h>

#include "cli.h"

#define MAXARGC 50
static struct cli_cmd_tuple *user_cmds;

/* Tokenize the blank chars from the string.
 * Works like the argc, argv generator. */
static int clitok (char *s, const char **argv)
{
	int argc = 0;
	int i, n = strlen (s);
	int last_isspace;
	int instring = 0;

	argv[0] = NULL;
	for (i=0, last_isspace=1; i<n+1; i++){
		if (s[i] == '"' && instring){
			instring = 0;
		}else if (s[i] == '"' && !instring){
			instring = 1;
			continue;
		}
		if (!instring
				&& (isspace(s[i])
				|| s[i] == '\0'
				|| s[i] == '"')){
			if (!last_isspace){
				if (argc >= MAXARGC) return -1;
				last_isspace = 1;
				argc++;
			}
			s[i] = '\0';
		}else if (last_isspace){
			argv[argc] = s + i;
			last_isspace = 0;
		}
	}
	return argc;
}

/* transform a string in a argc, *argv[] and call the function from the
 *  argv[0]. If no function is found, try to run the "error" errorfn function at
 *  the last  entry of the table, { NULL, errorfn }, If errorfn doesn't exist
 *  return NULL. */
int cli_call (char *s)
{
	int i;
	const char *argv[MAXARGC];
	int argc;
#ifdef SECURE
	memset (argv, 0, MAXARGC);
#endif
	argc = clitok(s,argv);

	if (argc <= 0) return -1;
	for (	i=0;	/* look for command at argv[0]. */
			user_cmds[i].s
			&& argv[0]
			&& (strcmp (user_cmds[i].s, argv[0]) != 0);
		i++ ){}

	if (user_cmds[i].cb == NULL)
		return -1;
	return user_cmds[i].cb (argc, argv);
}

struct cli_cmd_tuple *cli_register_tuple (struct cli_cmd_tuple *cmds)
{
	struct cli_cmd_tuple *old = user_cmds;
	user_cmds = cmds;
	return old;
}
