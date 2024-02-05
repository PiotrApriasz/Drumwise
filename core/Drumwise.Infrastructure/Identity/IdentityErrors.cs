using Drumwise.Application.Common.Errors;
using Microsoft.AspNetCore.Identity;

namespace Drumwise.Infrastructure.Identity;

public static class IdentityErrors
{
    public static IEnumerable<Error> UserNotFound => 
        Error.ApplicationError(new [] {ErrorCodes.Identity.UserNotFound});

    public static IEnumerable<Error> UpdatingUserError(IEnumerable<IdentityError> errors) =>
        Error.IdentityError(errors);

    public static IEnumerable<Error> AddingUserToRoleError(IEnumerable<IdentityError> errors) =>
        Error.IdentityError(errors);
}