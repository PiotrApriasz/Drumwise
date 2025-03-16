using Drumwise.Application.Common.Errors;

namespace Drumwise.Features.Homeworks;

public static class HomeworkErrors
{
    public static IEnumerable<Error> HomeworkNotFound =>
        Error.ApplicationError(new[] { ErrorCodes.Homework.HomeworkNotFound });
}