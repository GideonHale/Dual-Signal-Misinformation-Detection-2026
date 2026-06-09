import agents.ollama_agent as ag
from openai import OpenAI
from roles import role_titles, roles
from orderly_mad import orderly_mad
import agents.agent_presets as ap
import json
import os
import csv

def get_gat_score(iden):
    '''
    iden: the post id of the post to get the GAT score for (without the 't3_' prefix)
    returns: the GAT score of the post
    '''
    with open(f"datasets/justice_final_results.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if str(row[0]) == str('t3_' + iden):
                return float(row[2])
    # if it doesn't find the score, raise an exception
    raise(Exception(f"Could not find GAT score for post {iden}"))

def main():
    # Define our debaters
    agents = [ap.joseph, ap.steven, ap.benjamin, ap.christopher, ap.elijah]

    print("--- Multi-Agent Debate System ---")
    half = 0
    while half != 1 and half != 2:
        half = int(input('\nHalf (1 or 2): '))

    # take one post from matched_claims_llm_scores.jsonl
    print("\nLoading data from matched_claims_llm_scores.jsonl...")
    data = []
    file_name = 'matched_claims_llm_scores.jsonl'
    with open(f"datasets/{file_name}", "r") as f:
        for line in f:
            data.append(json.loads(line))
    n = len(data)
    if n == 0:
        print(f"No data found in {file_name}")
        return


    # create a csv file to store the results
    tested_post_ids = [] # for storing all the post_ids that have already been tested
    save_file = f"test_result{half}.csv"
    if not os.path.exists(f"results/{save_file}"):
        print(f'Creating new results ({save_file}) file...')
        with open(f"results/{save_file}", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["post_id", "score_1", "score_2", "score_3", "score_4", "score_5"])
    else:
        print(f'Results file ({save_file}) already exists...')
    
        # gather all the post_ids that have already been tested
        with open(f"results/{save_file}", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                tested_post_ids.append(row[0])


    # number of tests to run
    prompt = "\nEnter number of tests to run: "
    num_tests = int(input(prompt))

    # number of turns for debate
    prompt = "Enter number of turns for debate: "
    num_turns = int(input(prompt))

    # number of judging rounds
    prompt = 'Enter number of rounds of judging: '
    num_judge = int(input(prompt))

    # take the given half of the data and run the experiment five times on each entry
    print(int((half - 1) * (n / 2)), int(half * n / 2))
    for post_idx in range(int((half - 1) * (n / 2)), int(half * n / 2)):
        print(f"\n--- Post {post_idx+1} / {int(half * n / 2)} ---")
        
        # load all the data
        post = data[post_idx]
        iden = post["post_id"]
        if iden in tested_post_ids:
            print(f"Post {iden} has already been tested. Skipping...")
            continue

        title = post["post_title"]
        source_score = post["source_score"]
        missing_source_rate = post["missing_source_rate"]
        num_articles = post["num_articles"]
        num_unrated = post["num_unrated"]
        related_articles = post["related_articles"]
        formatted_related_articles = json.dumps(related_articles)

        # print(f"Getting GAT score for post {iden}...")
        try:
            gat_score = get_gat_score(str(iden))
            print('GAT score successfully retrieved')
        except Exception as e:
            print('Error:', e)
            gat_score = 0.5
            print("Using default GAT score of 0.5")
        
        # test five times
        test_results = []
        for test_num in range(1, num_tests + 1):

            print(f"\n--- Test {test_num} / {num_tests} ---")
            
            # run the debate
            print("--- Commencing Debate ---")
            discussion = (
                f"[DEBATE RULES]\n"
                f"1. You are participating in a debate about the fakeness of the following news article.\n"
                f"2. Give a clear verdict (a numerical integer score between completely fake at 0 and completely reliable at 5) and then a brief, one-paragraph explanation of this and in response to any previous responses as well.\n"
            
                f"[DESCRIPTION OF DATA FIELDS]\n"
                f"Headline: The title of the news article.\n"
                f"Source score [from 0.0 to 1.0]: the average of all sources of related articles found in the Adfontes reliability dataset.\n"
                f"Missing source rate [from 0.0 to 1.0]: the rate of sources returned from the RAG system that did not have a match in the reliability dataset.\n"
                f"Number of articles [from 0 to 30]: the number of articles returned from the RAG system that had semantically similar titles to the source headline.\n"
                f"Number of unrated articles [from 0 to 30]: the number of articles returned from the RAG system that were not found in the reliability dataset.\n"
                f"GAT score [from 0.0 to 1.0]: a topological second opinion that signals whether community interaction patterns confirm or contradict the RAG's textual assessment.\n"
                f"Related articles: Set of related articles.\n"
                f"Reliability score [from 0 to 64]: the reliability of the source of the article according to AskNews.\n"
                
                f"[NEWS ARTICLE FOR DEBATE]\n"
                f"Headline: {title}\n"
                f"Source score: {source_score}\n"
                f"GAT score: {gat_score}\n"
                f"Missing source rate: {missing_source_rate}\n"
                f"Number of articles: {num_articles}\n"
                f"Number of unrated articles: {num_unrated}\n"
                f"Related articles: {formatted_related_articles}\n"
            )

            final_transcript = orderly_mad(
                discussion,
                agents,
                num_turns,
                order="shuffle"
            )

            print("--- Debate Concluded ---")
            
            # Summarize the debate
            print("--- Summarizing Debate ---")
            summary = ap.summarizer.respond(final_transcript)
            final_transcript.append(summary)

            # print("--- Summary ---")
            # print(summary.content)

            print("--- Judging Debate ---")
            verdicts = []
            for post_idx in range(num_judge):
                print(f'[Judge {post_idx+1} / {num_judge}] thinking...')

                # Judge the debate
                verdict = ap.judge.respond(final_transcript)

                # print("\n--- Verdict ---")
                # print(verdict.content)

                # Extract the verdict just to like super make sure that we have an integer value
                extracted_verdict = ap.extractor.respond([verdict]).content[0]

                # test to see whether it's an integer from 0 to 5
                if not extracted_verdict.isdigit():
                    print('Error: the extracted score', extracted_verdict, 'is not an integer')
                    continue
                elif not (0 <= int(extracted_verdict) <= 5):
                    print('Error: the extracted score', extracted_verdict, 'is not between 0 and 5')
                    continue
                
                # print('Score:', extracted_verdict)
                verdicts.append(int(extracted_verdict))
            
            print('Verdicts:', verdicts)
            # Average the scores
            if len(verdicts) == 0:
                final_score = np.nan
            else:
                print(f'Successful extraction of {len(verdicts)} verdicts')
                final_score = sum(verdicts) / len(verdicts) / 5 # the 5 means it's a scale from 0 to 5
            print("Final Score:", final_score)
            test_results.append(final_score)

        # save the results
        print(f"Test results for post {iden}: {test_results}")
        try:
            with open(f"results/{save_file}", "a") as f:
                writer = csv.writer(f)
                row = [iden]
                for test_val in test_results:
                    row.append(test_val)
                writer.writerow(row)
            print(f'Results saved to {save_file}')
        except Exception as e:
            print('Error:', e)
            print("Could not save results to file")
        
        # print("We're stopping here for now")
        # return # for debugging purposes

if __name__ == "__main__":
    main()