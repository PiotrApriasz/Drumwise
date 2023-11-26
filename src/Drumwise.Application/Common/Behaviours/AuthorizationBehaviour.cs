using System.Reflection;
using Drumwise.Application.Common.Exceptions;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Security;
using MediatR;

namespace Drumwise.Application.Common.Behaviours;

public class AuthorizationBehaviour<TRequest, TResponse>(IUser user, IIdentityService identityService)
    : IPipelineBehavior<TRequest, TResponse>
    where TRequest : notnull
{
    public async Task<TResponse> Handle(TRequest request, RequestHandlerDelegate<TResponse> next, CancellationToken cancellationToken)
    {
        var authorizeAttributes = request.GetType().GetCustomAttributes<AuthorizeAttribute>().ToList();

        if (authorizeAttributes.Count != 0)
        {
            if (user.Id is null)
                throw new UnauthorizedAccessException();

            var authorizeAttributesWithRoles = authorizeAttributes
                .Where(a => !string.IsNullOrWhiteSpace(a.Roles)).ToList();

            if (authorizeAttributesWithRoles.Any())
            {
                var authorized = false;

                foreach (var roles in authorizeAttributesWithRoles.Select(a => a.Roles.Split(',')))
                {
                    foreach (var role in roles)
                    {
                        var isInRole = await identityService.IsInRoleAsync(user.Id, role.Trim());
                        if (isInRole)
                        {
                            authorized = true;
                            break;
                        }
                    }
                }

                if (!authorized)
                    throw new ForbiddenAccessException();
            }

            var authorizeAttributesWithPolicies = authorizeAttributes
                .Where(a => !string.IsNullOrWhiteSpace(a.Policy)).ToList();

            if (authorizeAttributesWithPolicies.Count != 0)
            {
                foreach (var policy in authorizeAttributesWithPolicies.Select(a => a.Policy))
                {
                    var authorized = await identityService.AuthorizeAsync(user.Id, policy);

                    if (!authorized)
                        throw new ForbiddenAccessException();
                }
            }
        }

        return await next();
    }
}