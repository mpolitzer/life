/** @file
 * @defgroup cli cli
 * @{
 **************************************************************************** */
/* TODO: implement a cli_fast_call variant that accepts sorted tuples. */
#ifndef CLI_H
#define CLI_H

/** cli function prototype all regitered functions must folow.
 **************************************************************************** */
typedef int (*cli_callback) (int argc, const char *argv[]);

/** struct that maps the command names and its functions callback together.
 **************************************************************************** */
struct cli_cmd_tuple {
	const char *s;		/**< argv[0], will trigger function call. */
	cli_callback cb;	/**< function to be called. */
};

/** stores the tuple cli_call will consult when parsing.
 * @param cmds - array of type cli_cmd_tuple.
 * @return the previous registered tuple, or NULL.
 **************************************************************************** */
struct cli_cmd_tuple *cli_register_tuple (struct cli_cmd_tuple *cmds);

/** parses the string and executes the command from the tuple on match.
 * @note that it will change 's' while parsing at cli_call.
 **************************************************************************** */
int cli_call (char *s);

#endif /* CLI_H */
/* @} */
